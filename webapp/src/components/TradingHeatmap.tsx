import { memo, useMemo, useState } from 'react'
import type { Trade } from '../types'

type DensityMode = 'trades' | 'pnl' | 'avgPnl'
type ColumnMode = '1hour' | '2hour' | '4hour' | 'session'

interface TradingHeatmapProps {
  trades: Trade[]
}

const DAYS_OF_WEEK = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

function getColumnCount(mode: ColumnMode): number {
  if (mode === '1hour') return 24
  if (mode === '2hour') return 12
  if (mode === '4hour') return 6
  return 4
}

function getColumnIndex(mode: ColumnMode, hour: number): number {
  if (mode === '1hour') return hour
  if (mode === '2hour') return Math.floor(hour / 2)
  if (mode === '4hour') return Math.floor(hour / 4)
  if (hour < 6) return 0
  if (hour < 12) return 1
  if (hour < 18) return 2
  return 3
}

function formatCellLabel(col: number, mode: ColumnMode): string {
  if (mode === '1hour') return col.toString().padStart(2, '0')
  if (mode === '2hour') return (col * 2).toString().padStart(2, '0')
  if (mode === '4hour') return (col * 4).toString().padStart(2, '0')
  return ['Asian', 'European', 'US', 'After'][col] || ''
}

function TradingHeatmap({ trades }: TradingHeatmapProps) {
  const [densityMode, setDensityMode] = useState<DensityMode>('pnl')
  const [columnMode, setColumnMode] = useState<ColumnMode>('1hour')

  const heatmapData = useMemo(() => {
    const cols = getColumnCount(columnMode)
    const labels = Array.from({ length: cols }, (_, col) => formatCellLabel(col, columnMode))

    if (!trades.length) {
      return {
        cols,
        labels,
        values: new Float64Array(7 * cols),
        counts: new Uint32Array(7 * cols),
        maxValue: 0,
        minValue: 0,
      }
    }

    const sums = new Float64Array(7 * cols)
    const counts = new Uint32Array(7 * cols)

    for (const trade of trades) {
      const date = new Date(trade.timestamp)
      if (Number.isNaN(date.getTime())) continue

      const row = date.getDay()
      const col = getColumnIndex(columnMode, date.getHours())
      const idx = row * cols + col
      sums[idx] += trade.profitLoss ?? 0
      counts[idx] += 1
    }

    const values = new Float64Array(7 * cols)
    let maxValue = -Infinity
    let minValue = Infinity

    for (let idx = 0; idx < values.length; idx += 1) {
      const count = counts[idx]
      if (densityMode === 'trades') {
        values[idx] = count
      } else if (densityMode === 'avgPnl') {
        values[idx] = count > 0 ? sums[idx] / count : 0
      } else {
        values[idx] = sums[idx]
      }

      if (count > 0) {
        if (values[idx] > maxValue) maxValue = values[idx]
        if (values[idx] < minValue) minValue = values[idx]
      }
    }

    if (maxValue === -Infinity) {
      maxValue = 0
      minValue = 0
    }

    return { cols, labels, values, counts, maxValue, minValue }
  }, [trades, densityMode, columnMode])

  const getCellColor = (value: number, count: number): { background: string; color: string } => {
    if (count === 0) return { background: 'rgba(60, 61, 64, 0.1)', color: '#68727d' }

    const absMax = Math.max(Math.abs(heatmapData.maxValue), Math.abs(heatmapData.minValue), 1)

    if (densityMode === 'trades') {
      const intensity = Math.min(1, value / absMax)
      const green = Math.floor(135 + intensity * 60)
      return {
        background: `rgb(16, ${green}, 103)`,
        color: intensity > 0.5 ? '#fff' : '#181c20',
      }
    }

    if (value > 0) {
      const intensity = Math.min(1, value / absMax)
      const green = Math.floor(122 + intensity * 100)
      return {
        background: `rgb(16, ${green}, 103)`,
        color: intensity > 0.5 ? '#fff' : '#181c20',
      }
    }

    if (value < 0) {
      const intensity = Math.min(1, Math.abs(value) / absMax)
      const red = Math.floor(180 + intensity * 75)
      const low = Math.max(0, Math.floor(40 - intensity * 30))
      return {
        background: `rgb(${red}, ${low}, ${low})`,
        color: intensity > 0.5 ? '#fff' : '#181c20',
      }
    }

    return { background: 'rgba(60, 61, 64, 0.15)', color: '#68727d' }
  }

  const formatValue = (value: number): string => {
    if (densityMode === 'trades') return value.toString()
    return value >= 0 ? `$${Math.round(value)}` : `-$${Math.round(Math.abs(value))}`
  }

  return (
    <div className="trading-heatmap-container">
      <div className="heatmap-header">
        <h3>📊 Trading Activity Heatmap</h3>
        <div className="heatmap-controls">
          <div className="control-group">
            <label>DENSITY:</label>
            <button className={densityMode === 'trades' ? 'active' : ''} onClick={() => setDensityMode('trades')} type="button">
              Trades
            </button>
            <button className={densityMode === 'pnl' ? 'active' : ''} onClick={() => setDensityMode('pnl')} type="button">
              PnL
            </button>
            <button className={densityMode === 'avgPnl' ? 'active' : ''} onClick={() => setDensityMode('avgPnl')} type="button">
              Avg PnL
            </button>
          </div>

          <div className="control-group">
            <label>COLUMNS:</label>
            <button className={columnMode === '1hour' ? 'active' : ''} onClick={() => setColumnMode('1hour')} type="button">
              1 Hour
            </button>
            <button className={columnMode === '2hour' ? 'active' : ''} onClick={() => setColumnMode('2hour')} type="button">
              2 Hour
            </button>
            <button className={columnMode === '4hour' ? 'active' : ''} onClick={() => setColumnMode('4hour')} type="button">
              4 Hour
            </button>
            <button className={columnMode === 'session' ? 'active' : ''} onClick={() => setColumnMode('session')} type="button">
              Session
            </button>
          </div>

          <div className="control-group">
            <label>ROWS:</label>
            <button className="active" type="button">
              Day of Week
            </button>
          </div>
        </div>
      </div>

      <div className="heatmap-grid-wrapper">
        <div className="heatmap-grid" style={{ gridTemplateColumns: `80px repeat(${heatmapData.cols}, 1fr)` }}>
          <div className="heatmap-cell header-cell"></div>
          {heatmapData.labels.map((label, colIdx) => (
            <div key={`header-${colIdx}`} className="heatmap-cell header-cell">
              {label}
            </div>
          ))}

          {DAYS_OF_WEEK.map((day, rowIdx) => (
            <div key={day} style={{ display: 'contents' }}>
              <div className="heatmap-cell row-label">{day}</div>
              {Array.from({ length: heatmapData.cols }, (_, colIdx) => {
                const idx = rowIdx * heatmapData.cols + colIdx
                const value = heatmapData.values[idx]
                const count = heatmapData.counts[idx]
                const colors = getCellColor(value, count)
                return (
                  <div
                    key={`${rowIdx}-${colIdx}`}
                    className="heatmap-cell data-cell"
                    style={{ background: colors.background, color: colors.color }}
                    title={`${day} ${heatmapData.labels[colIdx]}: ${count} trades, ${formatValue(value)}`}
                  >
                    {count > 0 ? formatValue(value) : ''}
                  </div>
                )
              })}
            </div>
          ))}
        </div>
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

export default memo(TradingHeatmap)
