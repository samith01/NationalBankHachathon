import { useEffect, useState } from 'react'
import './index.css'
import AnalysisPage from './AnalysisPage'
import { API_BASE_URL, analyzeTrading, getTrades, mapApiResponseToAnalysis, uploadTradingHistory } from './lib/api'
import type { AnalysisResult, SessionHistoryItem, TraderType } from './types'

const HISTORY_KEY = 'nb-bias-detector-history-v1'

const traderPalette: Record<TraderType, string> = {
  'Calm Trader': '#087A67',
  'Loss Averse Trader': '#C17A00',
  Overtrader: '#D93236',
  'Revenge Trader': '#8E2E1A',
}

type Page = 'home' | 'analysis'

function App() {
  const [page, setPage] = useState<Page>('home')
  const [isParsing, setIsParsing] = useState(false)
  const [tradesCount, setTradesCount] = useState(0)
  const [parseIssues, setParseIssues] = useState<string[]>([])
  const [history, setHistory] = useState<SessionHistoryItem[]>([])
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)

  useEffect(() => {
    const raw = localStorage.getItem(HISTORY_KEY)
    if (!raw) return
    try {
      const parsed = JSON.parse(raw) as SessionHistoryItem[]
      setHistory(parsed.slice(0, 8))
    } catch {
      localStorage.removeItem(HISTORY_KEY)
    }
  }, [])

  const handleFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsParsing(true)
    setParseIssues([])
    
    try {
      // Upload file directly to API
      const sessionId = await uploadTradingHistory(file)

      // Avoid waterfall: fetch both endpoints in parallel
      const [apiResponse, fetchedTrades] = await Promise.all([
        analyzeTrading(sessionId),
        getTrades(sessionId),
      ])
      
      // Map API response to frontend format
      const result = mapApiResponseToAnalysis(apiResponse, fetchedTrades)
      
      setTradesCount(fetchedTrades.length)
      setAnalysis(result)
      
      if (result) {
        setPage('analysis')
        window.scrollTo(0, 0)
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to analyze trades'
      setParseIssues([`API Error: ${errorMessage}. Backend URL: ${API_BASE_URL}`])
      setTradesCount(0)
      setAnalysis(null)
    } finally {
      setIsParsing(false)
      event.target.value = ''
    }
  }

  const saveSession = () => {
    if (!analysis) return
    const { trades: _trades, ...analysisForHistory } = analysis
    const newItem: SessionHistoryItem = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      traderType: analysis.traderType,
      tradesCount: tradesCount || analysis.metrics.totalTrades,
      analysis: analysisForHistory as AnalysisResult,
    }
    const updated = [newItem, ...history].slice(0, 8)
    setHistory(updated)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
  }

  const goHome = () => {
    setPage('home')
    setAnalysis(null)
    setTradesCount(0)
    setParseIssues([])
    window.scrollTo(0, 0)
  }

  const personaAccent = analysis ? traderPalette[analysis.traderType] : '#D6001C'

  return (
    <div className="app-shell" style={{ '--accent-color': personaAccent } as React.CSSProperties}>
      <a className="skip-link" href="#main-content">
        Skip to Main Content
      </a>
      <header className="nb-header">
        <div className="nb-topbar">
          <div className="nb-logo" aria-label="National Bank Bias Detector" onClick={goHome} style={{ cursor: 'pointer' }}>
            <span className="nb-logo-mark" aria-hidden="true" />
            <span>Sentinel</span>
          </div>
          <nav aria-label="Main" className="nb-mainnav">
            {page === 'home' ? (
              <a href="#trading-input"> </a>
            ) : (
              <>
                <a href="#behavioral-profile">Profile</a>
                <a href="#graphical-insights">Insights</a>
                <a href="#coaching-plan">Coaching</a>
                <a href="#saved-history">History</a>
              </>
            )}
          </nav>
          <div className="nb-topbar-right">
            <a href="#">About Us</a>
            <button className="nb-signin" type="button">
              Français
            </button>
          </div>
        </div>
      </header>
      <div className="texture" />

      {page === 'home' && (
        <>
          <section className="hero">
            <div className="hero-copy">
              <p className="eyebrow">Trading Bias Detector</p>
              <h1>Take Control of Trading Decisions</h1>
              <p>
                Upload trade history, detect behavioral bias using ML in seconds, and receive personalized coaching for
                calm, loss-averse, overtrading, and revenge-trading profiles.
              </p>
              <div className="pill-row">
                <span>ML-Powered Analysis</span>
                <span>Personalized Feedback</span>
                <span>Real-Time Results</span>
              </div>
            </div>
          </section>
          <main id="main-content" className="layout">
            <section id="trading-input" className="card controls">
              <h2>Trading History Input</h2>
              <div className="control-grid">
                <label className="upload-zone">
                  <span className="upload-kicker">Step 1</span>
                  <strong>{isParsing ? 'Reading File…' : 'Upload Trading File'}</strong>
                  <span>Drop a `.csv`, `.xls`, or `.xlsx` file with your trading history.</span>
                  <input
                    className="file-input"
                    type="file"
                    name="trade-file"
                    aria-label="Upload trade file"
                    accept=".csv,.xlsx,.xls"
                    onChange={handleFile}
                  />
                  <small>Required fields: timestamp, buy/sell, asset, quantity, entry/exit, P/L, account balance.</small>
                </label>
                <article className="upload-notes" aria-label="Upload notes">
                  <h3>Before You Upload</h3>
                  <ul>
                    <li>Use one trade per row.</li>
                    <li>Numbers can include decimals.</li>
                    <li>Missing values are flagged under Data Integrity Notes.</li>
                  </ul>
                  <p>
                    Example: 2025-03-01 9:30,NFLX,SELL,4,1754.20,1756.06,7.36,10007.36
                  </p>
                </article>
              </div>

              {parseIssues.length > 0 && (
                <ul className="issue-list" aria-live="polite">
                  {parseIssues.map((issue) => (
                    <li key={issue}>{issue}</li>
                  ))}
                </ul>
              )}
            </section>
          </main>
        </>
      )}

      {page === 'analysis' && analysis && (
        <AnalysisPage
          analysis={analysis}
          history={history}
          onBack={goHome}
          onSave={saveSession}
          onLoadHistory={(a) => setAnalysis(a)}
        />
      )}

      {isParsing && (
        <div className="loading-overlay">
          <div className="loading-container">
            <div className="spinner" />
            <h2 className="loading-title">Analyzing Your Trading Data</h2>
            <p className="loading-subtitle">Processing trades and detecting behavioral patterns...</p>
            <div className="progress-steps">
              <div className="progress-step active">
                <span className="step-number">1</span>
                <span className="step-label">Parsing File</span>
              </div>
              <div className="progress-step active">
                <span className="step-number">2</span>
                <span className="step-label">ML Analysis</span>
              </div>
              <div className="progress-step">
                <span className="step-number">3</span>
                <span className="step-label">Generating Report</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
