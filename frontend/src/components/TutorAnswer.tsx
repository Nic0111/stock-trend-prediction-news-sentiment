import { BookOpen, ExternalLink, MessageCircle, ShieldCheck } from "lucide-react";
import type { ChatMessage, ChatResponse, TutorResponse } from "../types";

/**
 * Convert the API envelope once so all tutor surfaces render the same
 * grounded answer, evidence chips and learner follow-ups.
 */
export function tutorMessageFromResponse(result: ChatResponse): ChatMessage {
  const tutor: TutorResponse = {
    message_id: result.message_id,
    mode: result.mode,
    direct_answer: result.direct_answer,
    explanation_steps: result.explanation_steps || [],
    evidence_refs: result.evidence_refs || [],
    key_terms: result.key_terms || [],
    caution: result.caution,
    follow_up_questions: result.follow_up_questions || [],
    model: result.model,
    fallback_used: result.fallback_used,
    context_id: result.context_id,
  };
  return {
    role: "assistant",
    id: result.message_id,
    content: result.content || result.direct_answer,
    tutor,
  };
}

function legacyTutorResponse(message: ChatMessage): TutorResponse | null {
  if (message.tutor || message.role !== "assistant" || !message.content.trim().startsWith("{")) return message.tutor || null;
  let parsed: Record<string, unknown> = {};
  try {
    const candidate = JSON.parse(message.content);
    if (candidate && typeof candidate === "object") parsed = candidate as Record<string, unknown>;
  } catch {
    const match = message.content.match(/"direct_answer"\s*:\s*"((?:\\.|[^"\\])*)"/s);
    if (match) parsed.direct_answer = match[1].replace(/\\"/g, '"').replace(/\\n/g, " ");
  }
  if (!parsed.direct_answer) return null;
  const mode = ["concept", "evidence", "news", "market", "uncertainty"].includes(String(parsed.mode))
    ? String(parsed.mode) as TutorResponse["mode"]
    : "evidence";
  return {
    message_id: message.id || "legacy-message",
    mode,
    direct_answer: String(parsed.direct_answer),
    explanation_steps: Array.isArray(parsed.explanation_steps) ? parsed.explanation_steps.map(String) : [],
    evidence_refs: [],
    key_terms: Array.isArray(parsed.key_terms)
      ? parsed.key_terms
        .filter((item): item is { term: string; definition: string } => Boolean(item && typeof item === "object" && "term" in item && "definition" in item))
        .map((item) => ({ term: String(item.term), definition: String(item.definition) }))
      : [],
    caution: parsed.caution ? String(parsed.caution) : null,
    follow_up_questions: Array.isArray(parsed.follow_up_questions) ? parsed.follow_up_questions.map(String) : [],
    model: "Previous tutor response",
    fallback_used: false,
    context_id: null,
  };
}

export function TutorAnswer({ message, onFollowUp }: { message: ChatMessage; onFollowUp: (question: string) => void }) {
  const tutor = legacyTutorResponse(message);
  if (!tutor) return <p className="tutor-legacy-copy">{message.content}</p>;
  return (
    <div className="tutor-answer">
      <p className="tutor-direct-answer">{tutor.direct_answer}</p>
      {tutor.explanation_steps.length > 0 && <p className="tutor-explanation">{tutor.explanation_steps.join(" ")}</p>}
      {tutor.evidence_refs.length > 0 && (
        <div className="tutor-evidence-refs">
          <span className="tutor-section-label"><ShieldCheck size={14} /> Evidence used</span>
          <div className="tutor-ref-list">
            {tutor.evidence_refs.map((reference) => (
              reference.url ? (
                <a className="tutor-ref" href={reference.url} target="_blank" rel="noreferrer" key={reference.id}>
                  <span>{reference.label}</span><ExternalLink size={12} />
                </a>
              ) : <span className="tutor-ref" key={reference.id}>{reference.label}</span>
            ))}
          </div>
        </div>
      )}
      {tutor.key_terms.length > 0 && (
        <div className="tutor-terms">
          <span className="tutor-section-label"><BookOpen size={14} /> In simple terms</span>
          {tutor.key_terms.map((term) => <details key={term.term}><summary>{term.term}</summary><p>{term.definition}</p></details>)}
        </div>
      )}
      {tutor.caution && <div className="tutor-caution"><strong>Keep this in context</strong><span>{tutor.caution}</span></div>}
      {tutor.follow_up_questions.length > 0 && (
        <div className="tutor-followups">
          <span className="tutor-section-label"><MessageCircle size={14} /> Continue learning</span>
          <div className="followup-list">{tutor.follow_up_questions.map((question) => <button type="button" key={question} onClick={() => onFollowUp(question)}>{question}</button>)}</div>
        </div>
      )}
    </div>
  );
}
