import type { AnalysisResult, Trade } from '../types'

export const API_BASE_URL = "https://1bxdwrf2-8000.use.devtunnels.ms";

async function readErrorDetails(response: Response): Promise<string> {
  const text = (await response.text()).trim()
  if (!text) return response.statusText
  try {
    const parsed = JSON.parse(text) as { detail?: string }
    return parsed.detail ?? text
  } catch {
    return text
  }
}

interface UploadResponse {
  session_id: string
  message: string
}

interface BiasDetectionResult {
  type: string
  confidence_score: number
  description: string
  recommendations: string[]
}

interface ApiAnalysisResponse {
  session_id: string
  biases_detected: BiasDetectionResult[]
  summary: {
    total_trades: number
    win_rate: number
    total_profit_loss: number
    primary_trader_type?: string
  }
}

const MAX_RENDER_TRADES = 4000
const MAX_CUMULATIVE_POINTS = 1200

function evenlySample<T>(items: T[], maxPoints: number): T[] {
  if (items.length <= maxPoints) return items
  const step = (items.length - 1) / (maxPoints - 1)
  const sampled: T[] = []
  for (let i = 0; i < maxPoints; i += 1) {
    sampled.push(items[Math.round(i * step)])
  }
  return sampled
}

export async function uploadTradingHistory(file: File): Promise<string> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/upload/trade-history`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const detail = await readErrorDetails(response)
    throw new Error(`Upload failed (${response.status}): ${detail}`)
  }

  const data: UploadResponse = await response.json()
  return data.session_id
}

export async function analyzeTrading(sessionId: string): Promise<ApiAnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/analyze/${sessionId}`)

  if (!response.ok) {
    const detail = await readErrorDetails(response)
    throw new Error(`Analysis failed (${response.status}): ${detail}`)
  }

  return response.json()
}

export async function getTrades(sessionId: string, signal?: AbortSignal): Promise<Trade[]> {
  const response = await fetch(`${API_BASE_URL}/data/${sessionId}`, { signal })

  if (!response.ok) {
    const detail = await readErrorDetails(response)
    throw new Error(`Failed to fetch trades (${response.status}): ${detail}`)
  }

  const data = await response.json()
  
  // Convert API format to frontend format
  return data.data.map((trade: any) => ({
    timestamp: trade.timestamp || new Date().toISOString(),
    side: (trade.side || 'BUY') as 'BUY' | 'SELL',
    asset: trade.asset || 'UNKNOWN',
    quantity: trade.quantity ?? null,
    entryPrice: trade.entry_price ?? null,
    exitPrice: trade.exit_price ?? null,
    profitLoss: trade.profit_loss ?? null,
    balance: trade.balance ?? null,
  }))
}

export function mapApiResponseToAnalysis(
  apiResponse: ApiAnalysisResponse,
  trades: Trade[]
): AnalysisResult {
  // Extract bias scores from API response
  const biasMap: Record<string, number> = {}
  apiResponse.biases_detected.forEach((bias) => {
    const normalizedType = bias.type.toLowerCase().replace(/\s+/g, '_')
    biasMap[normalizedType] = bias.confidence_score * 100 // Convert 0-1 to 0-100
  })

  const hourlyActivity = new Array<number>(24).fill(0)
  const cumulativePnLRaw: number[] = []
  let cumulative = 0
  let firstTimestamp = Number.NaN
  let lastTimestamp = Number.NaN
  let winCount = 0
  let winSum = 0
  let lossCount = 0
  let lossSum = 0

  for (const trade of trades) {
    const timestamp = new Date(trade.timestamp).getTime()
    if (Number.isFinite(timestamp)) {
      if (Number.isNaN(firstTimestamp)) firstTimestamp = timestamp
      lastTimestamp = timestamp
      hourlyActivity[new Date(timestamp).getHours()] += 1
    }

    const pnl = trade.profitLoss ?? 0
    if (pnl > 0) {
      winCount += 1
      winSum += pnl
    } else if (pnl < 0) {
      lossCount += 1
      lossSum += pnl
    }

    cumulative += pnl
    cumulativePnLRaw.push(cumulative)
  }

  const first = Number.isNaN(firstTimestamp) ? Date.now() : firstTimestamp
  const last = Number.isNaN(lastTimestamp) ? first + 60 * 60 * 1000 : lastTimestamp
  const tradingHours = Math.max((last - first) / (1000 * 60 * 60), 1)
  const tradesPerHour = trades.length / tradingHours
  const maxHourlyTrades = Math.max(...hourlyActivity)
  const winRate = trades.length > 0 ? (winCount / trades.length) * 100 : 0
  const averageWin = winCount > 0 ? winSum / winCount : 0
  const averageLoss = lossCount > 0 ? Math.abs(lossSum / lossCount) : 0
  const cumulativePnL = evenlySample(cumulativePnLRaw, MAX_CUMULATIVE_POINTS)
  const chartTrades = evenlySample(trades, MAX_RENDER_TRADES)

  // Determine primary trader type from highest scoring bias
  let primaryType: string = 'Calm Trader'
  let maxScore = biasMap['calm_trader'] ?? 0

  const typeMap: Record<string, string> = {
    'calm_trader': 'Calm Trader',
    'loss_averse_trader': 'Loss Averse Trader',
    'overtrader': 'Overtrader',
    'revenge_trader': 'Revenge Trader',
  }

  Object.entries(biasMap).forEach(([key, score]) => {
    if (score > maxScore) {
      maxScore = score
      primaryType = typeMap[key] ?? 'Calm Trader'
    }
  })

  // Get recommendations from the primary bias
  const primaryBias = apiResponse.biases_detected.find((bias) => 
    bias.type.toLowerCase().replace(/\s+/g, '_') === primaryType.toLowerCase().replace(/\s+/g, '_')
  )

  const recommendations = primaryBias?.recommendations ?? [
    'Continue monitoring your trading patterns',
    'Keep a trading journal to track emotional states',
    'Review performance metrics regularly',
  ]

  // Generate risk profile
  const riskScore = (biasMap['overtrader'] ?? 0) * 0.4 + (biasMap['revenge_trader'] ?? 0) * 0.6
  let riskLabel: 'Conservative' | 'Moderate' | 'Aggressive'
  if (riskScore < 30) {
    riskLabel = 'Conservative'
  } else if (riskScore < 60) {
    riskLabel = 'Moderate'
  } else {
    riskLabel = 'Aggressive'
  }

  return {
    biases: {
      overtrading: {
        score: biasMap['overtrader'] ?? 0,
        confidence: 85, // Using fixed confidence as API provides direct scores
        evidence: [`${tradesPerHour.toFixed(1)} trades/hour`, `${maxHourlyTrades} trades in busiest hour`],
      },
      lossAversion: {
        score: biasMap['loss_averse_trader'] ?? 0,
        confidence: 85,
        evidence: [`Win rate: ${winRate.toFixed(1)}%`, `Avg win: $${averageWin.toFixed(2)} vs avg loss: $${averageLoss.toFixed(2)}`],
      },
      revengeTrading: {
        score: biasMap['revenge_trader'] ?? 0,
        confidence: 85,
        evidence: [`Pattern analysis from ML model`, `Risk behavior after losses detected`],
      },
      calm: {
        score: biasMap['calm_trader'] ?? 0,
        confidence: 85,
        evidence: ['Disciplined trading pattern', 'Consistent risk management'],
      },
    },
    traderType: primaryType as any,
    recommendations,
    predictiveAlerts: riskScore > 70 ? [
      'High risk behavior detected - consider reducing position sizes',
      'Multiple bias patterns present - review your trading plan',
    ] : [],
    qualityIssues: [],
    riskProfile: {
      score: riskScore,
      label: riskLabel,
      rationale: `Based on ${tradesPerHour.toFixed(1)} trades/hour and behavioral patterns`,
    },
    portfolioSuggestion: riskScore > 60 
      ? 'Consider diversifying across multiple timeframes and reducing single-trade exposure'
      : 'Maintain current portfolio allocation with periodic rebalancing',
    metrics: {
      totalTrades: apiResponse.summary.total_trades || trades.length,
      winRate: apiResponse.summary.win_rate ?? winRate,
      averageWin,
      averageLoss,
      tradesPerHour,
      maxHourlyTrades,
      totalProfitLoss: apiResponse.summary.total_profit_loss ?? cumulativePnLRaw[cumulativePnLRaw.length - 1] ?? 0,
    },
    chartData: {
      cumulativePnL,
      hourlyActivity,
    },
    trades: chartTrades,
  }
}
