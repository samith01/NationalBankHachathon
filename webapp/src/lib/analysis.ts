import type { AnalysisResult, BiasResult, TraderType, Trade } from '../types'

const clamp = (value: number, min = 0, max = 100) => Math.min(max, Math.max(min, value))

const average = (values: number[]) => {
  if (!values.length) return 0
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

const toTime = (timestamp: string) => {
  const date = new Date(timestamp)
  return Number.isNaN(date.getTime()) ? null : date.getTime()
}

const qualityCheck = (trades: Trade[]) => {
  const issues: string[] = []
  const total = trades.length || 1
  const countMissing = (selector: (trade: Trade) => number | null) =>
    trades.filter((trade) => selector(trade) === null).length

  const missingQuantityPct = (countMissing((trade) => trade.quantity) / total) * 100
  const missingPnLPct = (countMissing((trade) => trade.profitLoss) / total) * 100
  const missingBalancePct = (countMissing((trade) => trade.balance) / total) * 100

  if (missingQuantityPct > 15) {
    issues.push('Quantity is missing in a large share of records; size-based signals are lower confidence.')
  }
  if (missingPnLPct > 10) {
    issues.push('P/L is missing in many rows; loss-aversion and revenge-trading signals are partially estimated.')
  }
  if (missingBalancePct > 25) {
    issues.push('Account balance is sparse; risk calibration against equity is approximate.')
  }

  const invalidTimes = trades.filter((trade) => toTime(trade.timestamp) === null).length
  if ((invalidTimes / total) * 100 > 5) {
    issues.push('Some timestamps are not parseable; activity clustering may be understated.')
  }
  return issues
}

const overtradingBias = (trades: Trade[]): BiasResult & { tradesPerHour: number; maxHourlyTrades: number } => {
  const timestamps = trades.map((trade) => toTime(trade.timestamp)).filter((value): value is number => value !== null)

  const first = timestamps[0] ?? Date.now()
  const last = timestamps[timestamps.length - 1] ?? first + 60 * 60 * 1000
  const tradingHours = Math.max((last - first) / (1000 * 60 * 60), 1)
  const tradesPerHour = trades.length / tradingHours

  const switches = trades.reduce((count, trade, index) => {
    if (index === 0) return count
    const previous = trades[index - 1]
    if (previous.asset === trade.asset && previous.side !== trade.side) {
      return count + 1
    }
    return count
  }, 0)

  const switchRate = trades.length > 1 ? switches / (trades.length - 1) : 0

  const hourlyActivity = new Array<number>(24).fill(0)
  trades.forEach((trade) => {
    const time = toTime(trade.timestamp)
    if (time !== null) {
      hourlyActivity[new Date(time).getHours()] += 1
    }
  })
  const maxHourlyTrades = Math.max(...hourlyActivity)

  const highImpactMoves = trades.reduce((count, trade, index) => {
    if (index === 0) return count
    const previous = trades[index - 1]
    const previousPnL = previous.profitLoss ?? 0
    const move = Math.abs(previousPnL)
    const severeMove = move > 250

    const prevTime = toTime(previous.timestamp)
    const nowTime = toTime(trade.timestamp)
    const quickReaction =
      prevTime !== null && nowTime !== null && nowTime - prevTime < 12 * 60 * 1000

    return severeMove && quickReaction ? count + 1 : count
  }, 0)

  const score = clamp(
    tradesPerHour * 6 + switchRate * 30 + Math.max(0, maxHourlyTrades - 8) * 4 +
      (highImpactMoves / Math.max(1, trades.length - 1)) * 30,
  )

  const confidence = clamp(60 + (timestamps.length / Math.max(1, trades.length)) * 40)

  return {
    score,
    confidence,
    evidence: [
      `${tradesPerHour.toFixed(1)} trades/hour over sampled history`,
      `${switches} side switches in same asset`,
      `${maxHourlyTrades} trades in busiest hour`,
    ],
    tradesPerHour,
    maxHourlyTrades,
  }
}

const lossAversionBias = (trades: Trade[]): BiasResult => {
  const wins = trades.map((trade) => trade.profitLoss).filter((value): value is number => (value ?? 0) > 0)
  const losses = trades.map((trade) => trade.profitLoss).filter((value): value is number => (value ?? 0) < 0)

  const avgWin = average(wins)
  const avgLoss = Math.abs(average(losses))
  const lossToWinRatio = avgWin > 0 ? avgLoss / avgWin : 2

  const winMoves: number[] = []
  const lossMoves: number[] = []

  for (const trade of trades) {
    if (trade.entryPrice === null || trade.exitPrice === null || trade.entryPrice === 0) {
      continue
    }
    const movePct = Math.abs(((trade.exitPrice - trade.entryPrice) / trade.entryPrice) * 100)
    if ((trade.profitLoss ?? 0) >= 0) {
      winMoves.push(movePct)
    } else {
      lossMoves.push(movePct)
    }
  }

  const avgWinMove = average(winMoves)
  const avgLossMove = average(lossMoves)
  const holdProxyRatio = avgWinMove > 0 ? avgLossMove / avgWinMove : 1.6

  const score = clamp(lossToWinRatio * 45 + holdProxyRatio * 20 + (losses.length > wins.length ? 15 : 0))
  const confidence = clamp(
    50 +
      Math.min(30, (wins.length + losses.length) / Math.max(1, trades.length) * 40) +
      Math.min(20, (winMoves.length + lossMoves.length) / Math.max(1, trades.length) * 20),
  )

  return {
    score,
    confidence,
    evidence: [
      `Avg win: $${avgWin.toFixed(2)} vs avg loss: $${avgLoss.toFixed(2)}`,
      `Loss-to-win ratio: ${lossToWinRatio.toFixed(2)}`,
      `Move ratio (losers vs winners): ${holdProxyRatio.toFixed(2)}`,
    ],
  }
}

const revengeBias = (trades: Trade[]): BiasResult => {
  let postLossSizeJump = 0
  let streakEscalations = 0
  let lossStreak = 0

  for (let i = 1; i < trades.length; i += 1) {
    const previous = trades[i - 1]
    const current = trades[i]

    const prevPnL = previous.profitLoss ?? 0
    const prevSize = (previous.quantity ?? 0) * (previous.entryPrice ?? 0)
    const currentSize = (current.quantity ?? 0) * (current.entryPrice ?? 0)

    if (prevPnL < 0 && prevSize > 0 && currentSize > prevSize * 1.25) {
      postLossSizeJump += 1
    }

    if (prevPnL < 0) {
      lossStreak += 1
    } else {
      lossStreak = 0
    }

    if (lossStreak >= 2 && prevSize > 0 && currentSize > prevSize * 1.1) {
      streakEscalations += 1
    }
  }

  const reactionRate = postLossSizeJump / Math.max(1, trades.length - 1)
  const streakRate = streakEscalations / Math.max(1, trades.length - 1)
  const score = clamp(reactionRate * 120 + streakRate * 180)
  const confidence = clamp(55 + Math.min(45, (trades.length / 120) * 45))

  return {
    score,
    confidence,
    evidence: [
      `${postLossSizeJump} larger-than-usual positions after losses`,
      `${streakEscalations} size escalations after loss streaks`,
      `Post-loss aggressive trade rate: ${(reactionRate * 100).toFixed(1)}%`,
    ],
  }
}

const riskProfile = (trades: Trade[]) => {
  const pnlSeries = trades.map((trade) => trade.profitLoss).filter((value): value is number => value !== null)
  const absAverage = average(pnlSeries.map((value) => Math.abs(value)))

  const balanceSeries = trades.map((trade) => trade.balance).filter((value): value is number => value !== null)
  let maxDrawdownPct = 0
  if (balanceSeries.length > 1) {
    let peak = balanceSeries[0]
    for (const balance of balanceSeries) {
      peak = Math.max(peak, balance)
      if (peak > 0) {
        const drawdown = ((peak - balance) / peak) * 100
        maxDrawdownPct = Math.max(maxDrawdownPct, drawdown)
      }
    }
  }

  const score = clamp(absAverage / 12 + maxDrawdownPct * 2)
  if (score >= 67) {
    return {
      score,
      label: 'Aggressive' as const,
      rationale: 'High notional swings and deep drawdown periods indicate elevated risk-taking behavior.',
    }
  }
  if (score >= 40) {
    return {
      score,
      label: 'Moderate' as const,
      rationale: 'Risk-taking is balanced but spikes appear during stress periods.',
    }
  }
  return {
    score,
    label: 'Conservative' as const,
    rationale: 'Position sizing and equity volatility remain controlled for most sessions.',
  }
}

const recommendationsFrom = (
  overtrading: BiasResult,
  lossAversion: BiasResult,
  revengeTrading: BiasResult,
  profileLabel: string,
) => {
  const recommendations: string[] = []
  const alerts: string[] = []

  if (overtrading.score > 50) {
    recommendations.push('Set a hard daily trade cap and lock execution after the cap to break impulse loops.')
    recommendations.push('Enable a 3-minute cooldown after each closed trade before opening the next position.')
    alerts.push('You are most likely to overtrade during your busiest one-hour window.')
  }
  if (lossAversion.score > 50) {
    recommendations.push('Define stop-loss and take-profit levels before entry and auto-attach bracket orders.')
    recommendations.push('Journal one sentence after each early winner to document whether the exit followed your plan.')
    alerts.push('A widening gap between average losses and average wins can trigger confidence erosion.')
  }
  if (revengeTrading.score > 45) {
    recommendations.push('After two consecutive losses, reduce position size by 40% for the next three trades.')
    recommendations.push('Use a 20-minute cooling-off rule before re-entry after a large loss.')
    alerts.push('Loss streaks are a trigger zone for oversized recovery attempts.')
  }

  if (!recommendations.length) {
    recommendations.push('Your trading profile is stable. Keep a weekly review cadence to preserve discipline.')
  }

  recommendations.push(
    `Risk profile is ${profileLabel.toLowerCase()}; align position limits and leverage with this profile to avoid hidden drift.`,
  )

  return { recommendations, alerts }
}

const traderTypeFrom = (
  overtrading: BiasResult,
  lossAversion: BiasResult,
  revengeTrading: BiasResult,
): TraderType => {
  const maxScore = Math.max(overtrading.score, lossAversion.score, revengeTrading.score)
  if (maxScore < 35) return 'Calm Trader'
  if (maxScore === lossAversion.score) return 'Loss Averse Trader'
  if (maxScore === revengeTrading.score) return 'Revenge Trader'
  return 'Overtrader'
}

export const analyzeTrades = (trades: Trade[]): AnalysisResult => {
  const overtrading = overtradingBias(trades)
  const lossAversion = lossAversionBias(trades)
  const revengeTrading = revengeBias(trades)

  const calmScore = clamp(100 - (overtrading.score * 0.4 + lossAversion.score * 0.35 + revengeTrading.score * 0.35))
  const calmBias: BiasResult = {
    score: calmScore,
    confidence: clamp(65 + trades.length / 8),
    evidence: ['Calm score rises when trade frequency, risk spikes, and recovery chasing stay controlled.'],
  }

  const traderType = traderTypeFrom(overtrading, lossAversion, revengeTrading)
  const profile = riskProfile(trades)
  const guidance = recommendationsFrom(overtrading, lossAversion, revengeTrading, profile.label)

  const wins = trades.map((trade) => trade.profitLoss ?? 0).filter((value) => value > 0)
  const losses = trades.map((trade) => trade.profitLoss ?? 0).filter((value) => value < 0)
  const pnlSequence = trades.map((trade) => trade.profitLoss ?? 0)
  const cumulativePnL: number[] = []
  let running = 0
  for (const value of pnlSequence) {
    running += value
    cumulativePnL.push(running)
  }

  const hourlyActivity = new Array<number>(24).fill(0)
  trades.forEach((trade) => {
    const time = toTime(trade.timestamp)
    if (time !== null) {
      hourlyActivity[new Date(time).getHours()] += 1
    }
  })

  const totalTimes = trades
    .map((trade) => toTime(trade.timestamp))
    .filter((time): time is number => time !== null)
  const spanHours =
    totalTimes.length > 1
      ? Math.max(1, (totalTimes[totalTimes.length - 1] - totalTimes[0]) / (1000 * 60 * 60))
      : 1

  return {
    biases: {
      overtrading,
      lossAversion,
      revengeTrading,
      calm: calmBias,
    },
    traderType,
    recommendations: guidance.recommendations,
    predictiveAlerts: guidance.alerts,
    qualityIssues: qualityCheck(trades),
    riskProfile: profile,
    portfolioSuggestion:
      profile.label === 'Aggressive'
        ? 'Shift 10-15% into lower-beta assets or cash buffer to stabilize emotional decision windows.'
        : 'Maintain current allocation and rebalance monthly to keep behavior aligned with your plan.',
    metrics: {
      totalTrades: trades.length,
      winRate: trades.length ? (wins.length / trades.length) * 100 : 0,
      averageWin: average(wins),
      averageLoss: Math.abs(average(losses)),
      tradesPerHour: trades.length / spanHours,
      maxHourlyTrades: Math.max(...hourlyActivity),
      totalProfitLoss: running,
    },
    chartData: {
      cumulativePnL,
      hourlyActivity,
    },
  }
}
