import Papa from 'papaparse'
import type { ParseOutcome, Trade } from '../types'

export const FIELD_LABELS: Record<string, string> = {
  timestamp: 'Timestamp',
  side: 'Side (Buy/Sell)',
  asset: 'Asset / Symbol',
  quantity: 'Quantity',
  entryPrice: 'Entry Price',
  exitPrice: 'Exit Price',
  profitLoss: 'Profit / Loss',
  balance: 'Account Balance',
}

export const REQUIRED_FIELDS = ['timestamp', 'side', 'asset'] as const

export const ALIASES = {
  timestamp: ['timestamp', 'time', 'date', 'datetime', 'executed_at'],
  side: ['side', 'buy_sell', 'action', 'direction', 'type'],
  asset: ['asset', 'symbol', 'ticker', 'instrument'],
  quantity: ['quantity', 'qty', 'size', 'amount', 'volume', 'shares'],
  entryPrice: ['entry_price', 'entry', 'price', 'open_price'],
  exitPrice: ['exit_price', 'exit', 'close_price'],
  profitLoss: ['profit_loss', 'p_l', 'pnl', 'profit', 'gain', 'realized_pnl'],
  balance: ['balance', 'account_balance', 'equity'],
} as const

export type ColumnMapping = Record<string, string>

export const normalizeHeader = (value: string) =>
  value
    .toLowerCase()
    .trim()
    .replace(/[\s-]+/g, '_')

const toNumber = (value: unknown): number | null => {
  if (value === undefined || value === null || value === '') {
    return null
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

const toSide = (value: unknown): 'BUY' | 'SELL' | null => {
  if (typeof value !== 'string') {
    return null
  }
  const normalized = value.trim().toLowerCase()
  if (['buy', 'b', 'long'].includes(normalized)) {
    return 'BUY'
  }
  if (['sell', 's', 'short'].includes(normalized)) {
    return 'SELL'
  }
  return null
}

export const buildHeaderMap = (headers: string[]) => {
  const normalizedHeaders = headers.map((header) => normalizeHeader(header))

  const findIndex = (keys: readonly string[]) => {
    for (let i = 0; i < normalizedHeaders.length; i += 1) {
      if (keys.includes(normalizedHeaders[i])) {
        return i
      }
    }
    return -1
  }

  return {
    timestamp: findIndex(ALIASES.timestamp),
    side: findIndex(ALIASES.side),
    asset: findIndex(ALIASES.asset),
    quantity: findIndex(ALIASES.quantity),
    entryPrice: findIndex(ALIASES.entryPrice),
    exitPrice: findIndex(ALIASES.exitPrice),
    profitLoss: findIndex(ALIASES.profitLoss),
    balance: findIndex(ALIASES.balance),
  }
}

/** Auto-detect column mapping from headers, returns field->headerName map */
export const autoDetectMapping = (headers: string[]): ColumnMapping => {
  const map = buildHeaderMap(headers)
  const result: ColumnMapping = {}
  for (const [field, idx] of Object.entries(map)) {
    if (idx >= 0) {
      result[field] = headers[idx]
    }
  }
  return result
}

/** Read just the headers from a file */
export const readFileHeaders = async (file: File): Promise<string[]> => {
  const fileName = file.name.toLowerCase()

  if (fileName.endsWith('.csv')) {
    const parsed = await new Promise<Papa.ParseResult<string[]>>((resolve, reject) => {
      Papa.parse<string[]>(file, {
        preview: 1,
        complete: (results) => resolve(results),
        error: (error) => reject(error),
      })
    })
    return (parsed.data[0] ?? []).map((cell) => String(cell ?? ''))
  }

  if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls')) {
    const XLSX = await import('xlsx')
    const buffer = await file.arrayBuffer()
    const workbook = XLSX.read(buffer, { type: 'array' })
    const firstSheet = workbook.Sheets[workbook.SheetNames[0]]
    const rows = XLSX.utils.sheet_to_json(firstSheet, {
      header: 1,
      blankrows: false,
    }) as unknown[][]
    return (rows[0] ?? []).map((cell) => String(cell ?? ''))
  }

  return []
}

/** Build header map from user-provided column mapping */
const buildHeaderMapFromMapping = (headers: string[], mapping: ColumnMapping) => {
  const result: Record<string, number> = {
    timestamp: -1,
    side: -1,
    asset: -1,
    quantity: -1,
    entryPrice: -1,
    exitPrice: -1,
    profitLoss: -1,
    balance: -1,
  }
  for (const [field, headerName] of Object.entries(mapping)) {
    const idx = headers.indexOf(headerName)
    if (idx >= 0) {
      result[field] = idx
    }
  }
  return result
}

const parseRows = (rows: unknown[][], headerMap: Record<string, number>): ParseOutcome => {
  const issues: string[] = []

  if (headerMap.timestamp === -1) {
    issues.push('Missing timestamp column.')
  }
  if (headerMap.side === -1) {
    issues.push('Missing buy/sell side column.')
  }
  if (headerMap.asset === -1) {
    issues.push('Missing asset column.')
  }

  const trades: Trade[] = []
  for (let i = 1; i < rows.length; i += 1) {
    const row = rows[i]
    if (!row || row.every((cell) => cell === null || cell === undefined || cell === '')) {
      continue
    }

    const timestamp =
      headerMap.timestamp >= 0 ? String(row[headerMap.timestamp] ?? '').trim() : ''
    const side = headerMap.side >= 0 ? toSide(row[headerMap.side]) : null
    const asset = headerMap.asset >= 0 ? String(row[headerMap.asset] ?? '').trim() : ''

    if (!timestamp || !side || !asset) {
      continue
    }

    trades.push({
      timestamp,
      side,
      asset,
      quantity: headerMap.quantity >= 0 ? toNumber(row[headerMap.quantity]) : null,
      entryPrice: headerMap.entryPrice >= 0 ? toNumber(row[headerMap.entryPrice]) : null,
      exitPrice: headerMap.exitPrice >= 0 ? toNumber(row[headerMap.exitPrice]) : null,
      profitLoss: headerMap.profitLoss >= 0 ? toNumber(row[headerMap.profitLoss]) : null,
      balance: headerMap.balance >= 0 ? toNumber(row[headerMap.balance]) : null,
    })
  }

  if (!trades.length) {
    issues.push('No valid trade rows found after parsing.')
  }

  return { trades, issues }
}

const fromRows = (rows: unknown[][]): ParseOutcome => {
  if (!rows.length) {
    return { trades: [], issues: ['No rows found in the file.'] }
  }
  const headers = rows[0].map((cell) => String(cell ?? ''))
  const headerMap = buildHeaderMap(headers)
  return parseRows(rows, headerMap)
}

const readAllRows = async (file: File): Promise<unknown[][] | null> => {
  const fileName = file.name.toLowerCase()

  if (fileName.endsWith('.csv')) {
    const parsed = await new Promise<Papa.ParseResult<string[]>>((resolve, reject) => {
      Papa.parse<string[]>(file, {
        complete: (results) => resolve(results),
        error: (error) => reject(error),
      })
    })
    return parsed.data
  }

  if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls')) {
    const XLSX = await import('xlsx')
    const buffer = await file.arrayBuffer()
    const workbook = XLSX.read(buffer, { type: 'array' })
    const firstSheet = workbook.Sheets[workbook.SheetNames[0]]
    return XLSX.utils.sheet_to_json(firstSheet, {
      header: 1,
      blankrows: false,
    }) as unknown[][]
  }

  return null
}

export const parseTradeFile = async (file: File): Promise<ParseOutcome> => {
  const rows = await readAllRows(file)
  if (!rows) {
    return { trades: [], issues: ['Unsupported file type. Upload CSV, XLS, or XLSX.'] }
  }
  return fromRows(rows)
}

export const parseTradeFileWithMapping = async (
  file: File,
  mapping: ColumnMapping,
): Promise<ParseOutcome> => {
  const rows = await readAllRows(file)
  if (!rows || !rows.length) {
    return { trades: [], issues: ['No rows found in the file.'] }
  }
  const headers = rows[0].map((cell) => String(cell ?? ''))
  const headerMap = buildHeaderMapFromMapping(headers, mapping)
  return parseRows(rows, headerMap)
}
