# StockSense Bursa news annotation guide

## What is being labelled

Each row represents one article associated with one of StockSense's ten Bursa
Malaysia companies. Read the title and description in `model_text`. Use the
recovered article text only as supporting context. The deployed model receives
the title and description, so the label must be defensible from that input.

Confirm `language=en` only when the model text is English. Do not translate a
non-English article and label the translation as if it were the archived text.

## Entity relevance

Set `entity_relevant=true` only when the article is materially about the
selected company or clearly discusses an event affecting it.

Set it to `false` when the match is caused by:

- an ambiguous abbreviation such as MISC;
- a passing company mention that is unrelated to the article's main event;
- a different overseas company with a similar name;
- a software package, registry page, directory or other non-news page;
- generic sector news with no defensible company-level connection.

Irrelevant rows remain useful for measuring the automatic entity filter, but
they do not receive a sentiment label and do not enter FinBERT training.

## Sentiment labels

Label the implication for the selected company's financial or operating
outlook. Do not predict the future share price.

- `positive`: the article describes a clear benefit, improvement or reduced
  risk, such as stronger earnings, material contract wins, improving margins,
  successful expansion or favourable regulatory outcomes.
- `negative`: the article describes a clear deterioration or added risk, such
  as weaker earnings, operational disruption, litigation, penalties, lost
  contracts, rising costs or an adverse outlook.
- `neutral`: the article is factual, balanced, immaterial, or contains no clear
  net implication for the selected company.

When positive and negative statements coexist, label the net company-level
implication supported by the text. Use `neutral` if the balance cannot be
resolved without speculation.

## Independent labels and adjudication

`annotator_1_label` and `annotator_2_label` must be entered independently.
Annotators must not see existing FinBERT probabilities before labelling.

- If both labels agree, leave `adjudicated_label` empty.
- If they disagree, a third reviewer records the final label in
  `adjudicated_label`.
- Validation and test rows require two labels.
- At least 25% of training rows require two labels.

Run `python main.py validate_bursa_news_dataset` after each annotation batch.
The validator resolves agreed/adjudicated labels, measures Cohen's kappa and
reports incomplete or inconsistent rows without changing the source file.

## Event categories

Choose the closest stable category:

- `earnings`
- `guidance`
- `contract`
- `corporate_action`
- `financing`
- `operations`
- `regulatory_legal`
- `management`
- `product_service`
- `macro_sector_exposure`
- `esg`
- `other`

The category is explanatory metadata, not a sentiment label.

## Leakage boundary

Never manually change `split`. It is assigned from publication time:

- 2015–2023: training;
- 2024: validation and calibration;
- 2025: frozen test.

Near-duplicate articles from the same event are assigned one `event_id` and
must not appear in more than one split.
