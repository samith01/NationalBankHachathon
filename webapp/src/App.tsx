import { useEffect, useState } from 'react'
import './index.css'
import AnalysisPage from './AnalysisPage'
import ColumnMapper from './ColumnMapper'
import { analyzeTrades } from './lib/analysis'
import { autoDetectMapping, parseTradeFileWithMapping, readFileHeaders } from './lib/parsers'
import type { ColumnMapping } from './lib/parsers'
import type { AnalysisResult, SessionHistoryItem, Trade, TraderType } from './types'

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
  const [trades, setTrades] = useState<Trade[]>([])
  const [parseIssues, setParseIssues] = useState<string[]>([])
  const [history, setHistory] = useState<SessionHistoryItem[]>([])
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)

  // Column mapping state
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [fileHeaders, setFileHeaders] = useState<string[]>([])
  const [detectedMapping, setDetectedMapping] = useState<ColumnMapping>({})

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
    try {
      const headers = await readFileHeaders(file)
      if (!headers.length) {
        setParseIssues(['Could not read headers from this file.'])
        return
      }
      const mapping = autoDetectMapping(headers)
      setUploadedFile(file)
      setFileHeaders(headers)
      setDetectedMapping(mapping)
      setParseIssues([])
    } catch {
      setParseIssues(['Failed to read file. Please check format and retry.'])
    } finally {
      setIsParsing(false)
      event.target.value = ''
    }
  }

  const handleMappingConfirm = async (mapping: ColumnMapping) => {
    if (!uploadedFile) return
    setIsParsing(true)
    try {
      const outcome = await parseTradeFileWithMapping(uploadedFile, mapping)
      setTrades(outcome.trades)
      setParseIssues(outcome.issues)
      const result = outcome.trades.length ? analyzeTrades(outcome.trades) : null
      setAnalysis(result)
      if (result) {
        setPage('analysis')
        setUploadedFile(null)
        setFileHeaders([])
        setDetectedMapping({})
        window.scrollTo(0, 0)
      }
    } catch {
      setParseIssues(['Failed to parse file. Please check format and retry.'])
      setTrades([])
      setAnalysis(null)
    } finally {
      setIsParsing(false)
    }
  }

  const handleMappingCancel = () => {
    setUploadedFile(null)
    setFileHeaders([])
    setDetectedMapping({})
  }

  const saveSession = () => {
    if (!analysis || !trades.length) return
    const newItem: SessionHistoryItem = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      traderType: analysis.traderType,
      tradesCount: trades.length,
      analysis,
    }
    const updated = [newItem, ...history].slice(0, 8)
    setHistory(updated)
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated))
  }

  const goHome = () => {
    setPage('home')
    setAnalysis(null)
    setTrades([])
    setParseIssues([])
    setUploadedFile(null)
    setFileHeaders([])
    setDetectedMapping({})
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
              <a href="#trading-input">Bias Detection</a>
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
              <p className="eyebrow">National Bank Bias Detector</p>
              <h1>Take Control of Trading Decisions</h1>
              <p>
                Upload trade history, detect behavioral bias in seconds, and receive personalized coaching for
                calm, loss-averse, overtrading, and revenge-trading profiles.
              </p>
              <div className="pill-row">
                <span>Behavioral Finance Grounded</span>
                <span>Personalized Feedback</span>
                <span>Fast Analysis</span>
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
                    This keeps the workflow simple and aligned with the challenge: file-based trading history input and
                    fast personalized analysis.
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

            {uploadedFile && (
              <ColumnMapper
                fileName={uploadedFile.name}
                fileSize={uploadedFile.size}
                headers={fileHeaders}
                initialMapping={detectedMapping}
                onConfirm={handleMappingConfirm}
                onCancel={handleMappingCancel}
              />
            )}
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
    </div>
  )
}

export default App
