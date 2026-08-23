import { describe, expect, it } from "vitest";
import { GLOSSARY_TERMS, LEARNING_MODULES } from "./learningCenterContent";

describe("learning center curriculum", () => {
  it("contains a complete, ordered field guide", () => {
    expect(LEARNING_MODULES.map((module) => module.id)).toEqual(["evidence", "market", "model", "news", "signals", "compare"]);
    expect(new Set(LEARNING_MODULES.map((module) => module.id)).size).toBe(LEARNING_MODULES.length);
    LEARNING_MODULES.forEach((module) => {
      expect(module.sections.length).toBeGreaterThanOrEqual(3);
      expect(module.takeaways.length).toBeGreaterThanOrEqual(2);
      expect(module.checks).toHaveLength(4);
      module.checks.forEach((check) => {
        expect(check.options.length).toBeGreaterThanOrEqual(3);
        expect(check.answer).toBeGreaterThanOrEqual(0);
        expect(check.answer).toBeLessThan(check.options.length);
      });
    });
  });

  it("keeps glossary terms mapped to a known lesson", () => {
    const moduleIds = new Set(LEARNING_MODULES.map((module) => module.id));
    expect(GLOSSARY_TERMS.length).toBeGreaterThanOrEqual(12);
    expect(new Set(GLOSSARY_TERMS.map((term) => term.term)).size).toBe(GLOSSARY_TERMS.length);
    GLOSSARY_TERMS.forEach((term) => expect(moduleIds.has(term.module)).toBe(true));
  });
});
