import type { AnalysisResult, Trade } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

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

export async function uploadTradingHistory(file: File): Promise<string> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/upload/trade-history`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`)
  }

  const data: UploadResponse = await response.json()
  return data.session_id
}

export async function analyzeTrading(sessionId: string): Promise<ApiAnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/analyze/${sessionId}`)

  if (!response.ok) {
    throw new Error(`Analysis failed: ${response.statusText}`)
  }

  return response.json()
}

export async function getTrades(sessionId: string): Promise<Trade[]> {
  const response = await fetch(`${API_BASE_URL}/data/${sessionId}`)

  if (!response.ok) {
    throw new Error(`Failed to fetch trades: ${response.statusText}`)
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

  // Calculate local metrics for visualization
  const timestamps = trades.map((trade) => new Date(trade.timestamp).getTime())
  const first = timestamps[0] ?? Date.now()
  const last = timestamps[timestamps.length - 1] ?? first + 60 * 60 * 1000
  const tradingHours = Math.max((last - first) / (1000 * 60 * 60), 1)
  const tradesPerHour = trades.length / tradingHours

  const hourlyActivity = new Array<number>(24).fill(0)
  trades.forEach((trade) => {
    const hour = new Date(trade.timestamp).getHours()
    hourlyActivity[hour] += 1
  })
  const maxHourlyTrades = Math.max(...hourlyActivity)

  const wins = trades.filter((t) => (t.profitLoss ?? 0) > 0)
  const losses = trades.filter((t) => (t.profitLoss ?? 0) < 0)
  const winRate = trades.length > 0 ? (wins.length / trades.length) * 100 : 0
  const averageWin = wins.length > 0 ? wins.reduce((sum, t) => sum + (t.profitLoss ?? 0), 0) / wins.length : 0
  const averageLoss = losses.length > 0 ? Math.abs(losses.reduce((sum, t) => sum + (t.profitLoss ?? 0), 0) / losses.length) : 0

  // Calculate cumulative P/L
  let cumulative = 0
  const cumulativePnL = trades.map((trade) => {
    cumulative += trade.profitLoss ?? 0
    return cumulative
  })

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
      totalTrades: trades.length,
      winRate,
      averageWin,
      averageLoss,
      tradesPerHour,
      maxHourlyTrades,
    },
    chartData: {
      cumulativePnL,
      hourlyActivity,
    },
    trades,  // Include trades for detailed visualization
  }
}
