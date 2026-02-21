---
name: nb-bias-detector-challenge
description: Builds solutions aligned with National Bank Bias Detector Challenge requirements, including data input, bias detection logic, personalization, and judging criteria.
---

# National Bank Bias Detector Challenge

## Purpose
Use this skill to implement or review features so they match the official challenge brief for the National Bank Bias Detector project.

## When to Use
- Defining scope for the Bias Detector prototype
- Designing CSV/Excel import and trade-entry workflows
- Implementing behavioral bias detection rules or models
- Generating user-facing feedback and recommendations
- Checking judging-readiness (performance, creativity, behavioral insight, personalization)

## Required Product Capabilities

### 1. Trading History Input (support one or more)
1. File upload for CSV/Excel trading records
2. Simple UI form for manual sample trades

**Expected core fields:**
- Timestamp
- Buy/Sell
- Asset
- Quantity
- Entry price
- Exit price
- P/L
- Account balance

### 2. Mandatory Bias Detection

#### Overtrading
Detect patterns such as:
- Excessive trade count relative to account balance
- Frequent position switching
- Trading shortly after large losses or wins
- Time-clustered overactivity (for example, too many trades in one hour)

#### Loss Aversion
Detect patterns such as:
- Letting losing trades run too long
- Closing winning trades too early
- Unbalanced risk/reward
- Average loss size larger than average win size

#### Revenge Trading
Detect patterns such as:
- Larger trade sizing immediately after losses
- Increased risk-taking after negative P/L streaks

### 3. Feedback and Recommendations
Output should include:
- Bias summaries in plain language
- Graphical insights (charts, timelines, heatmaps)
- Personalized recommendations, such as:
  - Daily trade limits
  - Stop-loss discipline
  - Cooling-off periods
  - Journaling prompts for trading psychology

## Optional Enhancements
- Detect additional behavioral biases
- Portfolio optimization suggestions
- Sentiment analysis on trader notes
- Risk profile scoring
- Predictive alerts for likely bias-triggering situations
- AI trading coach chatbot
- Stress/emotional state tagging

## Judging-Aligned Quality Checklist

### Performance
- Analysis is fast enough for practical use
- Design scales to larger datasets (target at least 20x mock-data size)

### Creativity
- UX and visualizations are intentional and engaging
- AI/ML or rules are used in meaningful, explainable ways

### Behavioral Finance Insight
- Bias definitions and signals are behaviorally sound
- Explanations are understandable to non-experts
- Interpretation goes beyond shallow metrics

### Personalization
- Feedback adapts to each trader's history
- Recommendations are specific, not generic
- Outputs update dynamically with new trade data

## Implementation Guardrails
- Prefer transparent, auditable signal logic over black-box claims
- Keep recommendations actionable and tied to observed behavior
- Separate detection confidence from recommendation severity
- Handle missing or dirty data gracefully and report data quality issues
