import { useMemo } from 'react'
import type { AnalysisResult, SessionHistoryItem } from './types'
import TradingHeatmap from './components/TradingHeatmap'

const round = (value: number) => Math.round(value * 100) / 100

const toPolyline = (values: number[]) => {
    if (!values.length) return ''
    const max = Math.max(...values)
    const min = Math.min(...values)
    const spread = Math.max(1, max - min)
    return values
        .map((value, index) => {
            const x = (index / Math.max(1, values.length - 1)) * 100
            const y = 100 - ((value - min) / spread) * 100
            return `${x},${y}`
        })
        .join(' ')
}

const formatDate = (timestamp: string) => {
    const date = new Date(timestamp)
    if (Number.isNaN(date.getTime())) return timestamp
    return date.toLocaleString()
}

interface AnalysisPageProps {
    analysis: AnalysisResult
    history: SessionHistoryItem[]
    onBack: () => void
    onSave: () => void
    onLoadHistory: (analysis: AnalysisResult) => void
}

export default function AnalysisPage({ analysis, history, onBack, onSave, onLoadHistory }: AnalysisPageProps) {
    const sparklinePath = useMemo(() => {
        const values = analysis.chartData.cumulativePnL
        if (values.length < 2) return ''
        return toPolyline(values)
    }, [analysis])

    return (
        <>
            <div className="analysis-topbar">
                <button className="back-btn" type="button" onClick={onBack}>
                    ← Back to Home
                </button>
                <h1 className="analysis-title">Analysis Results</h1>
                <button className="ghost" type="button" onClick={onSave}>
                    Save to Local History
                </button>
            </div>

            <main id="main-content" className="layout">
                <section id="behavioral-profile" className="card">
                    <h2>Behavioral Profile</h2>
                    <p className="profile-line">
                        Active Profile: <strong>{analysis.traderType}</strong> · Risk Profile: <strong>{analysis.riskProfile.label}</strong>
                    </p>

                    <div className="bias-grid">
                        {(
                            [
                                ['overtrading', 'Overtrading'],
                                ['lossAversion', 'Loss Aversion'],
                                ['revengeTrading', 'Revenge Trading'],
                                ['calm', 'Calm Discipline'],
                            ] as const
                        ).map(([key, label]) => (
                            <article className="bias-card" key={key}>
                                <p>{label}</p>
                                <h3>{round(analysis.biases[key].score)}</h3>
                                <small>Confidence: {round(analysis.biases[key].confidence)}%</small>
                                <ul>
                                    {analysis.biases[key].evidence.slice(0, 2).map((line) => (
                                        <li key={line}>{line}</li>
                                    ))}
                                </ul>
                            </article>
                        ))}
                    </div>
                </section>

                <section id="graphical-insights" className="card">
                    <h2>Graphical Insights</h2>
                    <div className="charts-grid">
                        <article className="chart-card">
                            <h3>Cumulative P/L Timeline</h3>
                            {sparklinePath ? (
                                <svg viewBox="0 0 100 100" className="sparkline" preserveAspectRatio="none">
                                    <polyline points={sparklinePath} />
                                </svg>
                            ) : (
                                <p className="empty-state">Need at least 2 records to render the timeline.</p>
                            )}
                        </article>
                    </div>
                </section>

                {analysis.trades && analysis.trades.length > 0 && (
                    <section id="trading-heatmap" className="card">
                        <TradingHeatmap trades={analysis.trades} />
                    </section>
                )}

                <section id="coaching-plan" className="card recommendations">
                    <h2>Personalized Coaching</h2>
                    <div className="two-column">
                        <div>
                            <h3>Action Plan</h3>
                            <ul>
                                {analysis.recommendations.map((tip) => (
                                    <li key={tip}>{tip}</li>
                                ))}
                            </ul>
                        </div>
                        <div>
                            <h3>Predictive Alerts</h3>
                            <ul>
                                {analysis.predictiveAlerts.length ? (
                                    analysis.predictiveAlerts.map((alert) => <li key={alert}>{alert}</li>)
                                ) : (
                                    <li>No urgent triggers detected for the latest dataset.</li>
                                )}
                            </ul>

                            <h3>Portfolio Optimization Suggestion</h3>
                            <p>{analysis.portfolioSuggestion}</p>

                            <h3>Data Integrity Notes</h3>
                            <ul>
                                {analysis.qualityIssues.length ? (
                                    analysis.qualityIssues.map((issue) => <li key={issue}>{issue}</li>)
                                ) : (
                                    <li>No major data quality problems detected.</li>
                                )}
                            </ul>
                        </div>
                    </div>
                </section>

                <section className="card metrics-row">
                    <article>
                        <p>Total trades</p>
                        <h3>{analysis.metrics.totalTrades}</h3>
                    </article>
                    <article>
                        <p>Win rate</p>
                        <h3>{round(analysis.metrics.winRate)}%</h3>
                    </article>
                    <article>
                        <p>Average win / loss</p>
                        <h3>
                            ${round(analysis.metrics.averageWin)} / ${round(analysis.metrics.averageLoss)}
                        </h3>
                    </article>
                    <article>
                        <p>Trades per hour</p>
                        <h3>{round(analysis.metrics.tradesPerHour)}</h3>
                    </article>
                    <article>
                        <p>Risk score</p>
                        <h3>{round(analysis.riskProfile.score)}</h3>
                    </article>
                </section>

                <section id="saved-history" className="card">
                    <h2>Saved Analysis History</h2>
                    {history.length === 0 ? (
                        <p className="empty-state">No saved sessions yet.</p>
                    ) : (
                        <div className="history-list">
                            {history.map((item) => (
                                <button
                                    key={item.id}
                                    className="history-item"
                                    type="button"
                                    onClick={() => onLoadHistory(item.analysis)}
                                >
                                    <strong>{item.traderType}</strong>
                                    <span>{item.tradesCount} trades</span>
                                    <span>{formatDate(item.createdAt)}</span>
                                </button>
                            ))}
                        </div>
                    )}
                </section>
            </main>
        </>
    )
}
