import { useState, useMemo } from 'react'
import type { Trade } from '../types'

type DensityMode = 'trades' | 'pnl' | 'avgPnl'
type ColumnMode = '1hour' | '2hour' | '4hour' | 'session'
type RowMode = 'dayOfWeek' | 'byDate'

interface HeatmapCell {
    value: number
    count: number
    label: string
}

interface TradingHeatmapProps {
    trades: Trade[]
}

const DAYS_OF_WEEK = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
const HOURS_24 = Array.from({ length: 24 }, (_, i) => i)

export default function TradingHeatmap({ trades }: TradingHeatmapProps) {
    const [densityMode, setDensityMode] = useState<DensityMode>('pnl')
    const [columnMode, setColumnMode] = useState<ColumnMode>('1hour')
    const [rowMode, setRowMode] = useState<RowMode>('dayOfWeek')

    const formatCellLabel = (col: number, mode: ColumnMode): string => {
        switch (mode) {
            case '1hour':
                return col.toString().padStart(2, '0')
            case '2hour':
                return `${(col * 2).toString().padStart(2, '0')}`
            case '4hour':
                return `${(col * 4).toString().padStart(2, '0')}`
            case 'session':
                return ['Asian', 'European', 'US', 'After'][col] || ''
        }
    }

    const heatmapData = useMemo(() => {
        if (!trades.length) return { data: [], maxValue: 0, minValue: 0 }

        const cellData = new Map<string, { sum: number; count: number }>()

        trades.forEach((trade) => {
            const date = new Date(trade.timestamp)
            if (Number.isNaN(date.getTime())) return

            const dayOfWeek = date.getDay()
            const hour = date.getHours()
            const pnl = trade.profitLoss ?? 0

            // Determine column based on mode
            let colKey: number
            switch (columnMode) {
                case '1hour':
                    colKey = hour
                    break
                case '2hour':
                    colKey = Math.floor(hour / 2)
                    break
                case '4hour':
                    colKey = Math.floor(hour / 4)
                    break
                case 'session':
                    // 4 sessions: Asian (0-6), European (6-12), US (12-18), After-hours (18-24)
                    if (hour < 6) colKey = 0
                    else if (hour < 12) colKey = 1
                    else if (hour < 18) colKey = 2
                    else colKey = 3
                    break
            }

            // Determine row based on mode
            let rowKey: number | string
            if (rowMode === 'dayOfWeek') {
                rowKey = dayOfWeek
            } else {
                // By date: YYYY-MM-DD
                rowKey = date.toISOString().split('T')[0]
            }

            const key = `${rowKey}-${colKey}`
            const existing = cellData.get(key) || { sum: 0, count: 0 }
            cellData.set(key, {
                sum: existing.sum + pnl,
                count: existing.count + 1,
            })
        })

        // Convert to array format
        const rows = rowMode === 'dayOfWeek' ? DAYS_OF_WEEK.length : Array.from(new Set(
            Array.from(cellData.keys()).map(k => k.split('-')[0])
        )).length

        let cols: number
        switch (columnMode) {
            case '1hour':
                cols = 24
                break
            case '2hour':
                cols = 12
                break
            case '4hour':
                cols = 6
                break
            case 'session':
                cols = 4
                break
        }

        const grid: HeatmapCell[][] = []
        let maxValue = -Infinity
        let minValue = Infinity

        if (rowMode === 'dayOfWeek') {
            for (let row = 0; row < 7; row++) {
                const rowData: HeatmapCell[] = []
                for (let col = 0; col < cols; col++) {
                    const key = `${row}-${col}`
                    const cell = cellData.get(key) || { sum: 0, count: 0 }
                    
                    let value: number
                    switch (densityMode) {
                        case 'trades':
                            value = cell.count
                            break
                        case 'pnl':
                            value = cell.sum
                            break
                        case 'avgPnl':
                            value = cell.count > 0 ? cell.sum / cell.count : 0
                            break
                    }

                    if (cell.count > 0) {
                        maxValue = Math.max(maxValue, value)
                        minValue = Math.min(minValue, value)
                    }

                    rowData.push({
                        value,
                        count: cell.count,
                        label: formatCellLabel(col, columnMode),
                    })
                }
                grid.push(rowData)
            }
        }

        return { data: grid, maxValue, minValue, rows: rowMode === 'dayOfWeek' ? DAYS_OF_WEEK : [] }
    }, [trades, densityMode, columnMode, rowMode])

    const getCellColor = (value: number, count: number): { background: string; color: string } => {
        if (count === 0) {
            return { background: 'rgba(60, 61, 64, 0.1)', color: '#68727d' }
        }

        const { maxValue, minValue } = heatmapData
        const absMax = Math.max(Math.abs(maxValue), Math.abs(minValue))

        if (densityMode === 'trades') {
            // Green gradient for trade count
            const intensity = Math.min(1, value / absMax)
            const green = Math.floor(135 + intensity * 60)
            return {
                background: `rgb(16, ${green}, 103)`,
                color: intensity > 0.5 ? '#fff' : '#181c20',
            }
        } else {
            // Red/Green for PnL
            if (value > 0) {
                const intensity = Math.min(1, value / absMax)
                const green = Math.floor(122 + intensity * 100)
                return {
                    background: `rgb(16, ${green}, 103)`,
                    color: intensity > 0.5 ? '#fff' : '#181c20',
                }
            } else if (value < 0) {
                const intensity = Math.min(1, Math.abs(value) / absMax)
                const red = Math.floor(180 + intensity * 75)
                return {
                    background: `rgb(${red}, ${Math.floor(40 - intensity * 30)}, ${Math.floor(40 - intensity * 30)})`,
                    color: intensity > 0.5 ? '#fff' : '#181c20',
                }
            } else {
                return { background: 'rgba(60, 61, 64, 0.15)', color: '#68727d' }
            }
        }
    }

    const formatValue = (value: number): string => {
        if (densityMode === 'trades') {
            return value.toString()
        } else {
            return value >= 0 ? `$${Math.round(value)}` : `-$${Math.round(Math.abs(value))}`
        }
    }

    return (
        <div className="trading-heatmap-container">
            <div className="heatmap-header">
                <h3>📊 Trading Activity Heatmap</h3>
                <div className="heatmap-controls">
                    <div className="control-group">
                        <label>DENSITY:</label>
                        <button
                            className={densityMode === 'trades' ? 'active' : ''}
                            onClick={() => setDensityMode('trades')}
                            type="button"
                        >
                            Trades
                        </button>
                        <button
                            className={densityMode === 'pnl' ? 'active' : ''}
                            onClick={() => setDensityMode('pnl')}
                            type="button"
                        >
                            PnL
                        </button>
                        <button
                            className={densityMode === 'avgPnl' ? 'active' : ''}
                            onClick={() => setDensityMode('avgPnl')}
                            type="button"
                        >
                            Avg PnL
                        </button>
                    </div>

                    <div className="control-group">
                        <label>COLUMNS:</label>
                        <button
                            className={columnMode === '1hour' ? 'active' : ''}
                            onClick={() => setColumnMode('1hour')}
                            type="button"
                        >
                            1 Hour
                        </button>
                        <button
                            className={columnMode === '2hour' ? 'active' : ''}
                            onClick={() => setColumnMode('2hour')}
                            type="button"
                        >
                            2 Hour
                        </button>
                        <button
                            className={columnMode === '4hour' ? 'active' : ''}
                            onClick={() => setColumnMode('4hour')}
                            type="button"
                        >
                            4 Hour
                        </button>
                        <button
                            className={columnMode === 'session' ? 'active' : ''}
                            onClick={() => setColumnMode('session')}
                            type="button"
                        >
                            Session
                        </button>
                    </div>

                    <div className="control-group">
                        <label>ROWS:</label>
                        <button
                            className={rowMode === 'dayOfWeek' ? 'active' : ''}
                            onClick={() => setRowMode('dayOfWeek')}
                            type="button"
                        >
                            Day of Week
                        </button>
                    </div>
                </div>
            </div>

            <div className="heatmap-grid-wrapper">
                {heatmapData.data.length > 0 ? (
                    <div className="heatmap-grid" style={{ gridTemplateColumns: `80px repeat(${heatmapData.data[0].length}, 1fr)` }}>
                        {/* Header row */}
                        <div className="heatmap-cell header-cell"></div>
                        {heatmapData.data[0].map((cell, colIdx) => (
                            <div key={`header-${colIdx}`} className="heatmap-cell header-cell">
                                {cell.label}
                            </div>
                        ))}

                        {/* Data rows */}
                        {heatmapData.data.map((row, rowIdx) => (
                            <>
                                <div key={`row-label-${rowIdx}`} className="heatmap-cell row-label">
                                    {heatmapData.rows[rowIdx]}
                                </div>
                                {row.map((cell, colIdx) => {
                                    const colors = getCellColor(cell.value, cell.count)
                                    return (
                                        <div
                                            key={`${rowIdx}-${colIdx}`}
                                            className="heatmap-cell data-cell"
                                            style={{
                                                background: colors.background,
                                                color: colors.color,
                                            }}
                                            title={`${heatmapData.rows[rowIdx]} ${cell.label}: ${cell.count} trades, ${formatValue(cell.value)}`}
                                        >
                                            {cell.count > 0 ? formatValue(cell.value) : ''}
                                        </div>
                                    )
                                })}
                            </>
                        ))}
                    </div>
                ) : (
                    <p className="empty-state">No trading data available for heatmap</p>
                )}
            </div>

            <div className="heatmap-legend">
                <span className="legend-item">
                    <span className="legend-box loss"></span> Loss
                </span>
                <span className="legend-item">
                    <span className="legend-box profit"></span> Profit
                </span>
            </div>
        </div>
    )
}
