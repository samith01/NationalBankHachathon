from __future__ import annotations

from typing import List, Optional, TypedDict

from pydantic import BaseModel

ScalarValue = str | float | int | bool | None
MetricValue = str | float | int
TradeRecord = dict[str, ScalarValue]


class MetricsRecord(TypedDict):
    total_trades: int
    win_rate: float
    total_profit_loss: float
    avg_profit_per_trade: float
    max_drawdown: float


class AnalysisSummaryRecord(TypedDict):
    total_trades: int
    win_rate: float
    total_profit_loss: float
    primary_trader_type: str


class BiasDetectionResult(BaseModel):
    type: str
    confidence_score: float
    description: str
    recommendations: List[str]


class AnalysisStoreRecord(TypedDict):
    biases_detected: list[BiasDetectionResult]
    total_trades: int
    win_rate: float
    primary_type: str


class TraderAnalysis(TypedDict):
    type: str
    confidence_score: float
    description: str
    recommendations: list[str]
    all_bias_scores: dict[str, float]


class UploadResponse(BaseModel):
    session_id: str
    message: str


class TradeEntry(BaseModel):
    timestamp: str
    asset: str
    side: str
    quantity: Optional[float] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    profit_loss: Optional[float] = None
    balance: Optional[float] = None


class DataResponse(BaseModel):
    session_id: str
    total_records: int
    data: List[TradeEntry]


class RangeDataResponse(DataResponse):
    date_range: dict[str, str]


class AnalysisResponse(BaseModel):
    session_id: str
    biases_detected: List[BiasDetectionResult]
    summary: AnalysisSummaryRecord


class ExcludeCriteria(BaseModel):
    assets: Optional[List[str]] = None
    date_range: Optional[dict[str, str]] = None
    min_loss_amount: Optional[float] = None
    max_loss_amount: Optional[float] = None
    trade_ids: Optional[List[int]] = None


class WhatIfRequest(BaseModel):
    exclude_criteria: Optional[ExcludeCriteria] = None
    output_format: str = "timeseries"


class BalancePoint(BaseModel):
    timestamp: str
    original_balance: float
    simulated_balance: float


class WhatIfTimeseriesResponse(BaseModel):
    session_id: str
    simulation_name: str
    original_final_balance: float
    simulated_final_balance: float
    balance_change: float
    balance_timeseries: List[BalancePoint]


class WhatIfFinalBalanceResponse(BaseModel):
    session_id: str
    simulation_name: str
    original_final_balance: float
    simulated_final_balance: float
    balance_improvement: float
    improvement_percentage: float


class SimulatedTradeEntry(TradeEntry):
    included_in_simulation: bool
    simulated_balance: float


class WhatIfFullDatasetResponse(BaseModel):
    session_id: str
    simulation_name: str
    original_trades: int
    included_trades: int
    excluded_trades: int
    dataset: List[SimulatedTradeEntry]


class WhatIfDownloadRequest(BaseModel):
    exclude_criteria: ExcludeCriteria
    report_format: str = "csv"


class BiasSummary(BaseModel):
    bias_type: str
    count: int
    percentage: float


class PerformanceMetrics(BaseModel):
    total_trades: int
    win_rate: float
    total_profit_loss: float
    avg_profit_per_trade: float
    max_drawdown: float


class MetricsResponse(BaseModel):
    session_id: str
    performance_metrics: PerformanceMetrics
    bias_summary: List[BiasSummary]
