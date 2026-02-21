import uuid
from datetime import date, datetime

import polars as pl
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from .analysis_service import (
    build_bias_detection_results,
    calculate_performance_metrics,
    predict_trader_type_analysis,
)
from .data_service import (
    dataframe_to_records,
    get_all_trades,
    get_last_numeric_value,
    get_trades_in_range,
    optional_float,
    parse_csv_file,
    parse_csv_file_with_summary,
    record_to_trade_entry,
    timestamp_to_iso,
    validate_required_columns,
)
from .schemas import (
    AnalysisResponse,
    BalancePoint,
    BiasSummary,
    CsvProcessingSummary,
    DataResponse,
    ExcludeCriteria,
    MetricsResponse,
    PerformanceMetrics,
    RangeDataResponse,
    SimulatedTradeEntry,
    UploadResponse,
    WhatIfDownloadRequest,
    WhatIfFinalBalanceResponse,
    WhatIfFullDatasetResponse,
    WhatIfRequest,
    WhatIfTimeseriesResponse,
)
from .simulation_service import (
    calculate_simulated_balances,
    generate_simulation_name,
    identify_excluded_trades,
)
from .state import analysis_results, uploaded_files

router = APIRouter()


@router.post(
    "/upload/trade-history",
    response_model=UploadResponse,
    summary="Upload Trading History",
    description="Upload a CSV file containing trading history for analysis",
)
async def upload_trade_history(file: UploadFile = File(...)):
    try:
        session_id = str(uuid.uuid4())
        content = await file.read()
        df, csv_summary = parse_csv_file_with_summary(
            content,
            source_name=file.filename or "uploaded.csv",
        )
        if csv_summary.error_message:
            raise HTTPException(status_code=400, detail=csv_summary.error_message)
        validate_required_columns(df)

        uploaded_files[session_id] = df.write_csv().encode("utf-8")
        return UploadResponse(
            session_id=session_id,
            message=f"Trade history uploaded successfully. Session ID: {session_id}",
            csv_summary=CsvProcessingSummary(
                status=csv_summary.status,
                source_name=csv_summary.source_name,
                empty_cells=csv_summary.empty_cells,
                quantity_fills=csv_summary.quantity_fills,
                entry_fills=csv_summary.entry_fills,
                exit_fills=csv_summary.exit_fills,
                profit_fixes=csv_summary.profit_fixes,
                balance_fixes=csv_summary.balance_fixes,
                warnings=csv_summary.warnings,
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(exc)}")


@router.get(
    "/data/{session_id}",
    response_model=DataResponse,
    summary="Get All Trading Data",
    description="Retrieve all trading records from the uploaded file for a given session",
)
async def get_all_trading_data(session_id: str):
    trades = get_all_trades(session_id)
    trade_entries = [record_to_trade_entry(trade) for trade in trades]
    return DataResponse(session_id=session_id, total_records=len(trade_entries), data=trade_entries)


@router.get(
    "/data/{session_id}/range",
    response_model=RangeDataResponse,
    summary="Get Trading Data by Date Range",
    description="Fetch trading records within a specified date range for a given session",
)
async def get_trading_data_by_range(
    session_id: str,
    start_date: date = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: date = Query(..., description="End date in YYYY-MM-DD format"),
):
    if start_date > end_date:
        raise HTTPException(
            status_code=400, detail="Start date must be before or equal to end date"
        )

    trades = get_trades_in_range(session_id, start_date, end_date)
    trade_entries = [record_to_trade_entry(trade) for trade in trades]
    return RangeDataResponse(
        session_id=session_id,
        total_records=len(trade_entries),
        date_range={"start": start_date.isoformat(), "end": end_date.isoformat()},
        data=trade_entries,
    )


@router.get(
    "/metrics/{session_id}",
    response_model=MetricsResponse,
    summary="Get Performance Metrics",
    description="Calculate key performance metrics for the trading history",
)
async def get_performance_metrics(session_id: str):
    if session_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Session ID not found")

    try:
        content = uploaded_files[session_id]
        df = parse_csv_file(content)
        validate_required_columns(df)

        metrics = calculate_performance_metrics(df)
        trader_analysis = predict_trader_type_analysis(df)
        trader_confidence = trader_analysis["confidence_score"]
        trader_type_str = trader_analysis["type"].replace("Trader Type: ", "")

        bias_summary = [
            BiasSummary(
                bias_type=trader_type_str,
                count=df.height,
                percentage=trader_confidence * 100,
            )
        ]

        return MetricsResponse(
            session_id=session_id,
            performance_metrics=PerformanceMetrics(**metrics),
            bias_summary=bias_summary,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(exc)}")


@router.post(
    "/what-if/{session_id}/simulate",
    summary="Run What-If Simulation",
    description="Calculate alternative balance history by excluding specified trades",
)
async def what_if_simulation(session_id: str, request: WhatIfRequest):
    if session_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Session ID not found")

    try:
        content = uploaded_files[session_id]
        df = parse_csv_file(content)
        validate_required_columns(df)

        exclude_criteria = request.exclude_criteria or ExcludeCriteria()
        exclude_indices = identify_excluded_trades(df, exclude_criteria)
        simulated_df = calculate_simulated_balances(df, exclude_indices)
        simulation_name = generate_simulation_name(exclude_criteria)

        if request.output_format == "timeseries":
            rows = dataframe_to_records(simulated_df)
            timeseries_data = [
                BalancePoint(
                    timestamp=timestamp_to_iso(row["timestamp"]),
                    original_balance=optional_float(row["balance"]) or 0.0,
                    simulated_balance=optional_float(row["simulated_balance"]) or 0.0,
                )
                for row in rows
            ]

            original_final_balance = get_last_numeric_value(df, "balance")
            simulated_final_balance = get_last_numeric_value(simulated_df, "simulated_balance")

            return WhatIfTimeseriesResponse(
                session_id=session_id,
                simulation_name=simulation_name,
                original_final_balance=original_final_balance,
                simulated_final_balance=simulated_final_balance,
                balance_change=simulated_final_balance - original_final_balance,
                balance_timeseries=timeseries_data,
            )

        if request.output_format == "final_balance":
            original_balance = get_last_numeric_value(df, "balance")
            simulated_balance = get_last_numeric_value(simulated_df, "simulated_balance")
            improvement = simulated_balance - original_balance
            improvement_pct = (improvement / original_balance * 100) if original_balance != 0 else 0

            return WhatIfFinalBalanceResponse(
                session_id=session_id,
                simulation_name=simulation_name,
                original_final_balance=original_balance,
                simulated_final_balance=simulated_balance,
                balance_improvement=improvement,
                improvement_percentage=round(improvement_pct, 2),
            )

        if request.output_format == "full_dataset":
            rows = dataframe_to_records(simulated_df)
            full_dataset = [
                SimulatedTradeEntry(
                    timestamp=timestamp_to_iso(row["timestamp"]),
                    asset=str(row["asset"]),
                    side=str(row["side"]),
                    quantity=optional_float(row["quantity"]),
                    entry_price=optional_float(row["entry_price"]),
                    exit_price=optional_float(row["exit_price"]),
                    profit_loss=optional_float(row["profit_loss"]),
                    balance=optional_float(row["balance"]),
                    included_in_simulation=bool(row["included_in_simulation"]),
                    simulated_balance=optional_float(row["simulated_balance"]) or 0.0,
                )
                for row in rows
            ]

            return WhatIfFullDatasetResponse(
                session_id=session_id,
                simulation_name=simulation_name,
                original_trades=df.height,
                included_trades=df.height - len(exclude_indices),
                excluded_trades=len(exclude_indices),
                dataset=full_dataset,
            )

        raise HTTPException(
            status_code=400,
            detail="Invalid output_format. Must be 'timeseries', 'final_balance', or 'full_dataset'",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"What-if simulation failed: {str(exc)}")


@router.post(
    "/what-if/{session_id}/download",
    summary="Download What-If Report",
    description="Generate downloadable report of what-if simulation",
)
async def download_what_if_report(session_id: str, request: WhatIfDownloadRequest):
    if session_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Session ID not found")

    try:
        content = uploaded_files[session_id]
        df = parse_csv_file(content)
        validate_required_columns(df)

        exclude_indices = identify_excluded_trades(df, request.exclude_criteria)
        simulated_df = calculate_simulated_balances(df, exclude_indices)

        if request.report_format == "csv":
            csv_data = simulated_df.write_csv()
            headers = {
                "Content-Disposition": 'attachment; filename="what_if_analysis.csv"',
                "Content-Type": "text/csv",
            }
            return Response(content=csv_data, headers=headers)

        if request.report_format == "xlsx":
            raise HTTPException(
                status_code=400,
                detail="XLSX export is not available in the Polars-only pipeline. Use CSV.",
            )

        raise HTTPException(
            status_code=400, detail="Invalid report_format. Must be 'csv' or 'xlsx'"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(exc)}")


@router.get(
    "/analyze/{session_id}",
    response_model=AnalysisResponse,
    summary="Analyze Trading Behavior",
    description="Analyze uploaded trading history for behavioral biases",
)
async def analyze_trading_history(session_id: str):
    if session_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="Session ID not found")

    try:
        content = uploaded_files[session_id]
        df = parse_csv_file(content)
        validate_required_columns(df)

        trader_type_analysis = predict_trader_type_analysis(df)
        all_bias_scores = trader_type_analysis["all_bias_scores"]
        biases_detected = build_bias_detection_results(all_bias_scores)

        analysis_results[session_id] = {
            "biases_detected": biases_detected,
            "total_trades": df.height,
            "win_rate": (
                df.filter(pl.col("profit_loss").cast(pl.Float64, strict=False) > 0).height
                / df.height
                if df.height > 0
                else 0.0
            ),
            "primary_type": trader_type_analysis["type"],
        }

        total_profit_loss = float(
            df.select(
                pl.col("profit_loss").cast(pl.Float64, strict=False).fill_null(0.0).sum()
            ).item()
        )
        win_count = df.filter(pl.col("profit_loss").cast(pl.Float64, strict=False) > 0).height

        return AnalysisResponse(
            session_id=session_id,
            biases_detected=biases_detected,
            summary={
                "total_trades": df.height,
                "win_rate": round(win_count / df.height * 100, 2) if df.height > 0 else 0,
                "total_profit_loss": round(total_profit_loss, 2) if df.height > 0 else 0,
                "primary_trader_type": trader_type_analysis["type"],
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


@router.get(
    "/health",
    summary="Health Check",
    description="Check if the API is running properly",
)
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
