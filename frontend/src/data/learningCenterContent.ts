export type LearningModuleId =
  | "evidence"
  | "market"
  | "model"
  | "news"
  | "signals"
  | "compare";

export interface LessonSection {
  heading: string;
  body: string;
}

export interface KnowledgeCheck {
  id: string;
  question: string;
  options: string[];
  answer: number;
  explanation: string;
}

export interface LearningModule {
  id: LearningModuleId;
  number: string;
  eyebrow: string;
  title: string;
  summary: string;
  accent: string;
  sections: LessonSection[];
  takeaways: string[];
  checks: KnowledgeCheck[];
  terms: string[];
  visual: "calendar" | "market" | "threshold" | "news" | "signals" | "compare";
  handoff: { label: string; path: string; prompt?: string };
}

export interface GlossaryTerm {
  term: string;
  definition: string;
  module: LearningModuleId;
}

export const LEARNING_MODULES: LearningModule[] = [
  {
    id: "evidence",
    number: "01",
    eyebrow: "Start with the evidence",
    title: "Know what the screen is actually saying.",
    summary: "Before reading a signal, learn its company, date, market session and horizon.",
    accent: "#1976d2",
    visual: "calendar",
    sections: [
      { heading: "A stock is a dated observation", body: "StockSense starts with the latest complete Bursa Malaysia market session. The observation date tells you when the input was measured; it is not automatically today." },
      { heading: "Trading days are not calendar days", body: "A 5-day horizon means five Bursa sessions. Weekends and market holidays are excluded, so the calendar span can be longer." },
      { heading: "KLCI is the market reference", body: "The FTSE Bursa Malaysia KLCI gives the system a broad Malaysian reference for relative performance, beta and correlation. It is context, not a forecast for the selected company." },
    ],
    takeaways: ["Always read the observation date first.", "A horizon starts from the observation date.", "Every result should be interpreted with its freshness and source boundary."],
    checks: [
      { id: "date", question: "A result is observed on 21 July and uses a 10-day horizon. What does the horizon describe?", options: ["Ten trading sessions after 21 July", "Ten calendar days before 21 July", "The last ten headlines"], answer: 0, explanation: "The forecast window starts after the latest usable observation and counts Bursa trading sessions." },
      { id: "klci", question: "Why is KLCI shown beside a company?", options: ["It guarantees the company will follow the index", "It provides broad Malaysian market context", "It replaces the company price"], answer: 1, explanation: "KLCI is a comparison context. It does not replace company-specific evidence or guarantee direction." },
      { id: "observation-freshness", question: "Why can the observation date be earlier than today's calendar date?", options: ["The system uses the latest complete, usable Bursa session", "The model chooses a random historical date", "The news provider decides the stock date"], answer: 0, explanation: "A result begins from the latest market row that passes the required completeness and feature checks. Weekends, holidays or incomplete inputs can make that date earlier than today." },
      { id: "session-count", question: "Which days count toward a 5-trading-day horizon?", options: ["Five consecutive calendar dates", "Five Bursa trading sessions", "Only days with positive returns"], answer: 1, explanation: "Trading horizons count Bursa sessions. Weekends and Bursa Malaysia market holidays do not consume a forecast day." },
    ],
    terms: ["KLCI", "Observation date", "Trading horizon", "Adjusted close"],
    handoff: { label: "Open Market Snapshot", path: "/market-snapshot" },
  },
  {
    id: "market",
    number: "02",
    eyebrow: "Read the market",
    title: "Turn a price chart into a readable story.",
    summary: "Price, activity, momentum and risk describe what has happened—not what must happen next.",
    accent: "#16884b",
    visual: "market",
    sections: [
      { heading: "Price shows the path", body: "Open, high, low and close describe one session. Adjusted close makes historical periods easier to compare when dividends or splits are present." },
      { heading: "Activity gives movement a setting", body: "Volume and the 20-day volume ratio show whether a move happened during busier or quieter trading than usual." },
      { heading: "Momentum and risk answer different questions", body: "RSI and MACD describe momentum. Volatility and ATR describe the typical size of movement. Neither one says whether the next move will be up or down by itself." },
    ],
    takeaways: ["A rising price is not the same as a low-risk price.", "Use moving averages to reduce daily noise.", "Read return, activity and risk together."],
    checks: [
      { id: "rsi", question: "What does an RSI above 70 most directly describe?", options: ["A higher recent momentum zone", "A guaranteed price fall", "The number of shares traded"], answer: 0, explanation: "RSI describes recent momentum. It can be useful context, but it is not a standalone buy or sell rule." },
      { id: "volume", question: "A volume ratio of 2.0× means…", options: ["The price will double", "Trading volume was about twice its recent average", "Volatility is zero"], answer: 1, explanation: "Volume ratio compares recent activity with its 20-day average; it does not predict the price change." },
      { id: "adjusted-close", question: "Why is adjusted close useful when reading a long price history?", options: ["It accounts for distributions such as dividends and stock splits", "It removes every losing session", "It guarantees all companies have the same price"], answer: 0, explanation: "Adjusted close improves historical comparability by accounting for corporate distributions and splits. It does not remove genuine market movement." },
      { id: "atr", question: "What does a higher ATR primarily tell a beginner?", options: ["Recent price movements have typically been larger", "The next movement must be upward", "The company traded more shares than KLCI"], answer: 0, explanation: "ATR measures the typical size of recent movement. It describes movement range, not the direction of the next return." },
    ],
    terms: ["Volume ratio", "Moving average", "RSI", "MACD", "Volatility", "ATR"],
    handoff: { label: "Practice in Market Snapshot", path: "/market-snapshot" },
  },
  {
    id: "model",
    number: "03",
    eyebrow: "Understand the model",
    title: "Probability is a signal, not a promise.",
    summary: "Learn how a probability becomes Up or Down, and why evidence status changes how cautiously you should read it.",
    accent: "#6e56cf",
    visual: "threshold",
    sections: [
      { heading: "Each horizon has its own model", body: "StockSense trains separate 1-day, 5-day and 10-day models because the useful patterns and uncertainty can differ across horizons." },
      { heading: "The threshold creates the label", body: "The model estimates probability_up. The locked decision threshold converts that value into Up or Down. The threshold is not always exactly 50%." },
      { heading: "Evidence status is a safety label", body: "Accepted means the horizon passed the frozen observed and prospective-shadow gates. Limited means some evidence is useful but a safety check is missing. Experimental means the directional conclusion is research-only." },
    ],
    takeaways: ["Probability is not certainty or investment advice.", "Read the threshold and the margin together.", "Experimental evidence should not be presented as a confident conclusion."],
    checks: [
      { id: "threshold", question: "The probability is 54% and the locked threshold is 51%. What label follows?", options: ["Up, because 54% is above 51%", "Down, because 54% is below 100%", "Unavailable, because probabilities must equal 50%"], answer: 0, explanation: "A directional label compares probability_up with the locked threshold, not with an assumption that every threshold is 50%." },
      { id: "status", question: "What should you do with an experimental direction?", options: ["Treat it as a guaranteed recommendation", "Read it as research evidence and inspect its limitations", "Delete all market evidence"], answer: 1, explanation: "Experimental results can be useful to study, but StockSense intentionally withholds a confident directional conclusion." },
      { id: "horizon-models", question: "Why can StockSense use a different model for each horizon?", options: ["Useful patterns and uncertainty can differ across 1, 5 and 10 trading days", "Every horizon must have a different company", "Longer horizons cannot use probabilities"], answer: 0, explanation: "Each horizon is a separate prediction task. Its useful features, class balance and uncertainty can differ, so it is evaluated independently." },
      { id: "probability-margin", question: "Probability Up is 52% and the locked threshold is 51%. What does the 1-point margin suggest?", options: ["A small directional margin that should be interpreted cautiously", "A 99% guarantee of an increase", "The result is automatically unavailable"], answer: 0, explanation: "The label is Up, but the estimate is only one percentage point above its cutoff. The small margin is evidence of limited separation, not certainty." },
    ],
    terms: ["Probability", "Decision threshold", "Calibration", "Balanced accuracy", "MCC", "Experimental"],
    handoff: { label: "Open Stock Outlook", path: "/stock-outlook", prompt: "Explain the stock model probability and threshold for this outlook in simple language." },
  },
  {
    id: "news",
    number: "04",
    eyebrow: "Read news evidence",
    title: "Follow the article trail before trusting the tone.",
    summary: "Relevance, source diversity, recency and FinBERT confidence all shape the news context.",
    accent: "#c18400",
    visual: "news",
    sections: [
      { heading: "Retrieval is only the first filter", body: "NewsAPI is the primary provider. Marketaux is a quota-aware fallback. Company aliases and entity disambiguation remove or review articles that only mention a word by coincidence." },
      { heading: "The context window follows the horizon", body: "The 1-day, 5-day and 10-day contexts use 7, 14 and 30 days respectively. Older articles contribute less through half-life weighting." },
      { heading: "FinBERT scores language, not price", body: "FinBERT estimates positive, neutral or negative tone and a confidence for the text. It does not estimate the probability that the stock will rise. Malaysian Bursa generalisation remains a separate validation requirement." },
    ],
    takeaways: ["One article is not broad evidence.", "Confidence and recency affect contribution weight.", "Unavailable sentiment is an honest evidence state, not a neutral score."],
    checks: [
      { id: "half-life", question: "What does a half-life of 15 days mean in the 10-day context?", options: ["The article disappears after 15 days", "The recency part of its weight halves after about 15 days", "The stock price halves after 15 days"], answer: 1, explanation: "Half-life controls recency decay. It does not delete an article or predict a price move." },
      { id: "finbert", question: "What does FinBERT confidence describe?", options: ["How confident the language classifier is about the article tone", "How likely the stock is to rise", "How many shares traded"], answer: 0, explanation: "FinBERT confidence belongs to the text classification step. It is not a stock-direction probability." },
      { id: "coverage", question: "Why should one relevant article usually be treated as limited news coverage?", options: ["One article may not represent broad or independently confirmed evidence", "Every article has neutral sentiment", "FinBERT cannot classify a single article"], answer: 0, explanation: "A single article can provide useful context, but it offers little source or event diversity. Coverage communicates that limitation separately from sentiment tone." },
      { id: "entity-filter", question: "An article contains a company abbreviation but discusses an unrelated software package. What should happen?", options: ["It should be excluded or reviewed by entity disambiguation", "It should automatically receive positive sentiment", "It should replace the company market data"], answer: 0, explanation: "Keyword overlap is not enough. Entity disambiguation checks whether the article genuinely refers to the selected company before it can influence sentiment." },
    ],
    terms: ["FinBERT", "Sentiment confidence", "Coverage", "Independent event", "Half-life", "Entity disambiguation"],
    handoff: { label: "Inspect News & Sentiment", path: "/news", prompt: "Show me which articles contribute to this horizon's sentiment and explain why." },
  },
  {
    id: "signals",
    number: "05",
    eyebrow: "Connect the evidence",
    title: "See how signals agree—or disagree.",
    summary: "The aggregate panel is a transparent relationship between stock evidence and news evidence, not a magic score.",
    accent: "#df4b66",
    visual: "signals",
    sections: [
      { heading: "Reinforcing is not proof", body: "When stock direction and news tone point the same way, the relationship is reinforcing. It can improve context, but agreement does not make the result certain." },
      { heading: "Conflicting signals are informative", body: "A Down stock signal with positive news—or the reverse—means the evidence streams disagree. The disagreement should be visible, not averaged away." },
      { heading: "Fusion is gated", body: "StockSense keeps a stock-led synthesis unless the stock horizon, news coverage and a separately evaluated fusion artifact all pass their gates. A balanced or unavailable news context may leave the stock probability unchanged." },
    ],
    takeaways: ["The parent result can be traced back to two child signals.", "Balanced news does not mean positive news.", "A combined probability appears only when validated fusion is available."],
    checks: [
      { id: "relationship", question: "Stock trend is Down and news tone is Positive. What is the clearest description?", options: ["Conflicting signals", "Guaranteed recovery", "News is automatically ignored"], answer: 0, explanation: "The two evidence streams point in different directions, so the UI should expose that conflict for inspection." },
      { id: "fusion", question: "Why might the aggregate probability equal the stock probability?", options: ["The news panel is decorative", "News may be unavailable, low coverage, or fusion may not be validated", "The model cannot read numbers"], answer: 1, explanation: "StockSense preserves a transparent stock-led result when the evidence needed for numerical fusion is not ready." },
      { id: "reinforcing", question: "Stock trend is Up and usable news tone is Positive. How should their relationship be described?", options: ["Reinforcing evidence", "Conflicting evidence", "A guaranteed profit"], answer: 0, explanation: "Both evidence streams point in the same direction, so they reinforce one another. Agreement can strengthen context without guaranteeing the outcome." },
      { id: "balanced-tone", question: "What does balanced news tone contribute to an aggregate interpretation?", options: ["No material support for either direction", "A guaranteed Up adjustment", "Proof that every article is neutral"], answer: 0, explanation: "Balanced tone means the weighted news evidence does not materially favour positive or negative direction. It may leave a stock-led result unchanged." },
    ],
    terms: ["Aggregate result", "Stock-led synthesis", "Numerical fusion", "Reinforcing", "Conflicting", "Balanced"],
    handoff: { label: "Trace an Outlook", path: "/stock-outlook", prompt: "Explain how the stock trend and news sentiment relate in this outlook." },
  },
  {
    id: "compare",
    number: "06",
    eyebrow: "Compare responsibly",
    title: "Make a fair comparison before forming a view.",
    summary: "Compare companies using the same dates, horizon and evidence definitions—not just the largest percentage.",
    accent: "#168c9a",
    visual: "compare",
    sections: [
      { heading: "Normalize before comparing", body: "A normalized chart starts each company at the same base so you can compare the journey rather than the absolute share price." },
      { heading: "Return and risk belong together", body: "A higher return may come with higher volatility or weaker evidence. Look across price performance, activity, risk, model status and news coverage." },
      { heading: "Comparison is for learning", body: "StockSense does not rank companies or make buy/sell recommendations. Use the comparison to ask better questions and investigate the evidence behind differences." },
    ],
    takeaways: ["Use the same horizon for every company.", "Check dates and evidence status before comparing.", "A chart supports a question; it does not make the decision for you."],
    checks: [
      { id: "normalize", question: "Why do comparison charts start companies at the same base?", options: ["To compare relative movement fairly", "To hide the original prices", "To make every company equally risky"], answer: 0, explanation: "Normalizing makes the relative journey comparable even when the share prices have different starting levels." },
      { id: "risk", question: "Two companies have the same return. What should you inspect next?", options: ["Risk, activity, dates and evidence quality", "Only the company logo", "Nothing—the result is identical"], answer: 0, explanation: "Equal return does not mean equal experience or evidence. Risk and freshness change the interpretation." },
      { id: "fair-window", question: "What is required for a fair comparison between two companies?", options: ["Use the same observation period and prediction horizon", "Choose whichever date makes each company look strongest", "Compare only their latest absolute prices"], answer: 0, explanation: "A shared period and horizon keep the comparison aligned. Different windows can make performance and model evidence misleading." },
      { id: "normalized-limit", question: "What can a normalized return chart not tell you by itself?", options: ["The companies' original share prices and complete risk profile", "Which relative path rose more", "Whether both series started at the same base"], answer: 0, explanation: "Normalization highlights relative movement. You still need price, volatility, activity and evidence quality to understand the wider comparison." },
    ],
    terms: ["Normalized return", "Risk", "Evidence quality", "Benchmark", "Correlation"],
    handoff: { label: "Open Compare Tickers", path: "/compare" },
  },
];

export const GLOSSARY_TERMS: GlossaryTerm[] = [
  { term: "Accepted", definition: "A model evidence status showing that the horizon passed the required frozen evaluation gates.", module: "model" },
  { term: "Adjusted close", definition: "A historical closing price adjusted for distributions such as dividends and stock splits.", module: "market" },
  { term: "Aggregate result", definition: "The top-level interpretation that shows how stock-model evidence and usable news evidence relate.", module: "signals" },
  { term: "ATR", definition: "Average True Range; a measure of the typical size of recent price movement.", module: "market" },
  { term: "Balanced accuracy", definition: "Average recall across the Up and Down classes, useful when classes are uneven.", module: "model" },
  { term: "Benchmark", definition: "A reference result used to judge whether a model or company comparison adds useful information.", module: "compare" },
  { term: "Beta", definition: "A measure of how much a company has moved relative to a reference such as KLCI.", module: "market" },
  { term: "Calibration", definition: "Whether predicted probabilities match how often outcomes actually occur.", module: "model" },
  { term: "Candlestick", definition: "A chart mark showing one session's open, high, low and close prices.", module: "market" },
  { term: "Conflicting signals", definition: "A relationship where the stock direction and news tone point in different directions.", module: "signals" },
  { term: "Correlation", definition: "A measure of how closely two return series have moved together, without proving that one caused the other.", module: "compare" },
  { term: "Coverage", definition: "A summary of how much relevant, usable and diverse news evidence is available.", module: "news" },
  { term: "Decision threshold", definition: "The locked probability cutoff used to turn a probability into Up or Down.", module: "model" },
  { term: "Evidence status", definition: "The Accepted, Limited or Experimental label that tells you how cautiously to use a model result.", module: "model" },
  { term: "Entity disambiguation", definition: "The filtering step that checks whether an article truly refers to the selected company.", module: "news" },
  { term: "Experimental", definition: "Research-only evidence that has not passed all frozen acceptance gates.", module: "model" },
  { term: "FinBERT", definition: "A financial-language model used to classify article tone and confidence.", module: "news" },
  { term: "Freshness", definition: "How recent the latest usable market observation or news evidence is when the result is produced.", module: "evidence" },
  { term: "Half-life", definition: "The time it takes for a recency weight to fall to half its starting value.", module: "news" },
  { term: "Independent event", definition: "A distinct news event after duplicate or closely repeated coverage has been grouped together.", module: "news" },
  { term: "KLCI", definition: "A broad Bursa Malaysia index used as market context.", module: "evidence" },
  { term: "Limited", definition: "An evidence status indicating that some information is useful but one or more safety checks are not fully satisfied.", module: "model" },
  { term: "MACD", definition: "A momentum indicator based on the relationship between faster and slower moving averages.", module: "market" },
  { term: "MCC", definition: "Matthews correlation coefficient; a balanced measure of binary classification quality.", module: "model" },
  { term: "Moving average", definition: "An average price over a rolling window that reduces day-to-day noise and helps show the broader path.", module: "market" },
  { term: "News tone", definition: "The positive, neutral or negative language classification assigned to relevant financial news.", module: "news" },
  { term: "Normalized return", definition: "A return chart that starts each series at the same base for fair comparison.", module: "compare" },
  { term: "Numerical fusion", definition: "A separately validated calculation that combines stock and news inputs into an adjusted probability.", module: "signals" },
  { term: "OHLC", definition: "Open, high, low and close: the four core prices that summarize one market session.", module: "market" },
  { term: "Observation date", definition: "The date of the latest complete market row used to construct a prediction.", module: "evidence" },
  { term: "Percentage point", definition: "The direct difference between two percentages; for example, 54% is 3 percentage points above 51%.", module: "model" },
  { term: "Probability", definition: "The model's estimated chance of an Up outcome over the selected horizon.", module: "model" },
  { term: "Probability margin", definition: "The distance between a model probability and its locked decision threshold.", module: "model" },
  { term: "Recency weight", definition: "A multiplier that reduces an older article's influence in the selected news context.", module: "news" },
  { term: "Reinforcing signals", definition: "A relationship where the stock direction and usable news tone point the same way.", module: "signals" },
  { term: "ROC-AUC", definition: "A metric measuring how well a model ranks Up observations above Down observations across possible thresholds.", module: "model" },
  { term: "RSI", definition: "A momentum indicator comparing recent upward and downward closes.", module: "market" },
  { term: "Sentiment confidence", definition: "How confident the language classifier is in an article's tone label.", module: "news" },
  { term: "Shadow evaluation", definition: "A prospective check on newer unseen observations used to monitor whether model evidence remains dependable.", module: "model" },
  { term: "Source diversity", definition: "The number and variety of independent publishers supporting the available news context.", module: "news" },
  { term: "Stock-led synthesis", definition: "An aggregate interpretation that keeps the stock-model probability unchanged while describing how news agrees, conflicts or remains unavailable.", module: "signals" },
  { term: "Trading day", definition: "A Bursa Malaysia market session; weekends and exchange holidays are excluded.", module: "evidence" },
  { term: "Trading horizon", definition: "The number of future Bursa sessions covered by a forecast.", module: "evidence" },
  { term: "Volatility", definition: "A measure of how widely recent returns have moved around their average.", module: "market" },
  { term: "Volume", definition: "The number of shares traded during a market session.", module: "market" },
  { term: "Volume ratio", definition: "Current or recent volume divided by its rolling average, used to compare activity with normal levels.", module: "market" },
];
