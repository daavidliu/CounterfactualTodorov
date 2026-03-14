# PROMPT BUILDER & EVALUATION RUBRICS
# This module defines the prompt templates and Todorov-based evaluation rubrics 
# used for both Narrativity and Overall alignment experiments.

# --- Response Formatting Templates ---
INPUT = """[Story 1]: {premise} {initial} {original_ending}
Task: You must replace "{initial}" with "{counterfactual}" to create a [Story 2].
"""

INPUT_REWRITE_ONLY = """[Story 2]: {premise} {counterfactual} (TO BE COMPLETED...)
Task: Finish the story.
"""

# Full context story representation for the evaluator
OUTPUT = """[Story 2]: {premise} {counterfactual} {new_ending}"""

# --- Evaluator Persona & Core Rules ---
SELENE_PROMPT = """You are a logic engine tasked with evaluating a response. Your job is to verify structural and logical criteria. 

Here are some rules of the evaluation:
(1) **Style Agnosticism:** Do NOT evaluate the quality, length, creativity, or "depth" of the writing. A response that is short, boring, or simple MUST receive a Score of {max_score} if it satisfies the structural/logical criteria in the rubric.
(2) **The Subtraction Method:** Assume the response is a Score {max_score} by default. Only assign a lower score if you can quote specific text that violates the rubric. "Lack of detail" or "Room for Improvement" is NOT a violation.
(3) **Evidence:** In your reasoning, you must explicitly identify the parts of the text that trigger the rubric conditions when assigning a score.

Your reply should strictly follow this format:
**Result:** <an integer between 1 and {max_score}>
**Reasoning:** <Justify the Result using the checklist in the rubric>

Here is the data:

Instruction:
```
{input}
```

Response:
```
{output}
```

Score Rubrics:
[{criteria}]
{rubric}
"""

# --- Rubric Formatting Templates ---
SCORE_RUBRIC_TEMPLATE = """
Score 1: {score1_description}
Score 2: {score2_description}
Score 3: {score3_description}
""".strip()

SCORE_RUBRIC_TEMPLATE_FULL = """
Score 1: {score1_description}
Score 2: {score2_description}
Score 3: {score3_description}
Score 4: {score4_description}
Score 5: {score5_description}
""".strip()

# --- Narrative Alignment Criteria (1-3 Scale) ---
CRITERIA_OVERALL = """Story 2 is evaluated by the combination of three criteria: Conceivability, Coherence (Rational Connectivity), and Structure (Variance Subsumption).

1. Conceivability: Story 2 must depict a world without internal contradiction (can be unrealistic, unlikely, or fictional/fantastical).
2. Todorov Rational Connectivity: Story 2 must present interpretable Todorov narrative stages (Equilibrium, Disruption, Recognition, Attempt, New Equilibrium) that are contextually/causally connected. Causal connections can be explicit or reasonably inferred. 
3. Todorov Narrative Variance Subsumption: The *KEY Todorov stages (*Disruption, *Attempt to Resolve, *New Equilibrium) found in Story 1 must also be interpretable in Story 2. Story 2 does NOT need to subsume non-key stages (Equilibrium, Recognition) to satisfy this criteria."""

CRITERIA_CONCEIVABILITY = "Conceivability: Story 2 depicts a world without internal contradiction. (can be unrealistic, unlikely, or fictional/fantastical)"

CRITERIA_COHERENCE = """Todorov Rational Connectivity: If Story 2 presents multiple interpretable types of Todorov narrative stage(s) (Equilibrium - status quo, Disruption - inciting incident, Recognition of disruption, Attempt to Resolve, action in response to disruption, New Equilibrium - changed status quo): each stage is a consequence of, or a reaction to, a previous stage. Stages are contextually relevant and make sense.
Presumption of Rational Connection: If a causal connection is not explicit but can be reasonably inferred by a human reader, count it as connected.
Strict Logical Interpretation: Evaluate ONLY the causal logic and contextual relevance. Do NOT evaluate the quality, depth, engagement, or character motivations. A story that is short, simple, or boring MUST receive full marks if the causal logic holds together."""

CRITERIA_STRUCTURE = "To evaluate this criteria you must identify all types of abstract Todorov narrative stages that can be interpreted from both stories: Equilibrium - status quo, *Disruption - inciting incident, Recognition of disruption, *Attempt to Resolve, action in response to disruption, *New Equilibrium - changed status quo. (* indicate KEY stages) EVAULUATE THE CRITERIA - Todorov Narrative Variance Subsumption: Story 2 may be similar, identical, related, unrelated or incompatable to story 1; as long as the types of abstract Todorov structural stages which exist in story 1 also exist in story 2, this criteria is satisfied."

RUBRIC_OVERALL = SCORE_RUBRIC_TEMPLATE.format(
    score1_description="Story 2 fails to satisfy ONE or MORE of the three criteria:\n- Conceivability Failure: It contains internal contradictions.\n- Coherence Failure: Todorov stages are disconnected, nonsensical, reality-breaking, or contain non-sequiturs.\n- Structure Failure: Story 2 is missing *KEY Todorov stages (*Disruption, *Attempt, *New Equilibrium) that were present in Story 1.",
    score2_description="Evaluation is inconclusive due to ambiguity in ANY of the criteria.\nIt is unclear whether Story 2 is internally contradictory (Conceivability), logically connected (Coherence), or structurally complete regarding Story 1's *KEY stages (Structure).",
    score3_description="Story 2 successfully satisfies ALL three criteria:\n- Conceivable: No internal contradictions (unrealistic/fantasy elements are permitted).\n- Coherent: Todorov stages are rationally connected and contextually relevant (simple/short stories qualify).\n- Structural Subsumption: Story 2 contains ALL *KEY Todorov stages (*Disruption, *Attempt, *New Equilibrium) that exist in Story 1. (Story 2 may omit Equilibrium or Recognition stages even if present in Story 1).",
)

RUBRIC_CONCEIVABILITY = SCORE_RUBRIC_TEMPLATE.format(
    score1_description="Depicts any inconceivable elements or scenarios that contain internal contradictions to the story's logic.",
    score2_description="Conceivability is ambiguous. It could be interpreted as either conceivable or inconceivable.",
    score3_description="Depicts conceivable scenarios which may be unrealistic, unlikely, or fictional/fantastical. As long as there is no internal contradiction, the story satisfies conceivability.",
)

RUBRIC_COHERENCE = SCORE_RUBRIC_TEMPLATE.format(
    score1_description="Assign Score 1 if ANY of these failures exist:\n- The Todorov stages are NOT causally connected (even assuming the Presumption of Coherence).\n- There is nonsensical repetition.\n- There is irrationality in the story's logic and connection between Todorov stages.\n- There is a complete non-sequitur. \n- There is a REALITY-BREAKING contradiction in the stages (e.g., characters do somethingimpossible according to the story's logic).",
    score2_description="Assign Score 2 ONLY if it is unclear whether the Todorov Stages can be rationally interpreted from the story. It is HIGHLY AMBIGUOUS whether Todorov Stages are causally connected or contextaully relevant. (Do NOT use this score for simple or short stories that are logically sound. If the logic is rational, it must be Score 3).",
    score3_description="Assign Score 3 if ALL of the following are true:\n- Todorov stage(s) can be rationally interpreted.\n- There are NO reality-breaking contradictions or non-sequiturs.\n\nThen only if MULTIPLE stages exist:\n- Todorov stages are causally connected (explicitly or via reasonable inference).\n- Todorov stages are contextually relevant.\n\nIMPORTANT: If the connection/relevancy holds, you MUST assign Score 3. Do NOT downgrade to Score 2 or 1 because the story is 'simple', 'lacks detail', or 'needs more depth'. Simplicity is NOT a connectivity failure.",
)

RUBRIC_STRUCTURE = SCORE_RUBRIC_TEMPLATE.format(
    score1_description="Story 2's structure does not subsume ALL *KEY stages present in story 1. Story 1 has at least one *KEY Todorov stage (disruption, attempt to resolve, new equilibrium) absent from story 2, and as a result story 2 feels less complete as a narrative.",
    score2_description="It is unclear whether story 2's structure subsumes story 1's Todorov Structure. There is ambiguity/confusion when identifying Todorov structural stages and the criteria could be interpreted either way.",
    score3_description="Story 2's structure includes ALL *KEY stages present in story 1 and is equivalently complete or MORE. Story 2 may present stages in a different order. Story 2 does not need to present *KEY Todorov stages if they are also missing from story 1. Story 2 may miss the initial equilibrium stage or the recognition stage even if they exist in story 1.",
)

# --- Primary Narrativity Evaluation (1-5 Scale) ---
CRITERIA_NARRATIVITY = """To evaluate this criteria you must identify all types of abstract Todorov narrative stages present in the story: Equilibrium - status quo, *Disruption - inciting incident, Recognition of disruption, *Attempt to Resolve, action in response to disruption, *New Equilibrium - changed status quo. (* indicate KEY stages).
Identification Rules:
1) A single sentence, statement, or even a word CAN contain multiple interpretable narrative stages simultaneously.
2) Non-Sequential Rule: Stages do NOT need to happen in order or be complete. If a story skips a stage (e.g. goes straight from Disruption to New Equilibrium), the present stages are still VALID and CONSIDERED interpretable.
3) Strict Structural Interpretation: Do NOT evaluate the quality, detail, or length of the writing. A stage is "interpretable" if the logical event occurs, regardless of how vague or brief it is. (e.g., "He got mad" is a valid Recognition; "He left" is a valid New Equilibrium)."""

RUBRIC_NARRATIVITY = SCORE_RUBRIC_TEMPLATE_FULL.format(
    score1_description="Assign Score 1 if ANY of the following are true:\n- The Disruption stage is MISSING (and the story is NOT the specific 'Eq + New Eq' case described in Score 2).\n- Only the Equilibrium stage is present.\n- No Todorov stages are interpretable.",
    score2_description="Assign Score 2 if:\n- The ONLY stage present is Disruption.\n- OR the ONLY stages present are Equilibrium AND New Equilibrium.",
    score3_description="Assign Score 3 if:\n- 2 or more types of Todorov stages are interpretable.\n- The stages MUST include Disruption.",
    score4_description="Assign Score 4 if:\n- 3 or more types of Todorov stages are interpretable.\n- The stages MUST include Disruption.",
    score5_description="Assign Score 5 if:\n- 4 or more types of Todorov stages are interpretable.\n- The stages MUST include Disruption.\n\nIMPORTANT: If these conditions are met, you MUST assign Score 5. Do NOT downgrade to Score 4 because the story is 'vague', 'lacks detail', or 'could be improved'. If the stages are structurally present, the score is 5.",
)

def build_eval_prompt(premise, initial, original_ending, counterfactual, new_ending, model_name, evaluation_task):
    """
    Constructs a formatted prompt for the evaluation model.
    
    Args:
        premise: The premise shared by both story versions.
        initial: The initial event from Story 1.
        original_ending: Story 1's full ending.
        counterfactual: The counterfactual intervention point.
        new_ending: The model-generated ending for Story 2.
        model_name: Identifier for formatting (e.g., 'prometheus' or 'selene').
        evaluation_task: Rubric selector ('overall', 'plausibility', 'coherence', 'structure', 'narrativity').
    """
    
    # 1. Format the data representation based on the task requirements
    # 'Narrativity', 'Coherence', and 'Plausibility' usually just need Story 2
    if evaluation_task in ['plausibility', 'coherence', 'narrativity']:
        input_str = INPUT_REWRITE_ONLY.format(
            premise=premise,
            counterfactual=counterfactual
        )
    else:
        # 'Structure' and 'Overall' require comparison with Story 1
        input_str = INPUT.format(
            premise=premise,
            initial=initial,
            original_ending=original_ending,
            counterfactual=counterfactual
        )
    
    output_str = OUTPUT.format(
        premise=premise,
        counterfactual=counterfactual,
        new_ending=new_ending
    )
    
    # 2. Select appropriate rubric and max score
    task_map = {
        'overall': (CRITERIA_OVERALL, RUBRIC_OVERALL, 3),
        'plausibility': (CRITERIA_CONCEIVABILITY, RUBRIC_CONCEIVABILITY, 3),
        'coherence': (CRITERIA_COHERENCE, RUBRIC_COHERENCE, 3),
        'structure': (CRITERIA_STRUCTURE, RUBRIC_STRUCTURE, 3),
        'narrativity': (CRITERIA_NARRATIVITY, RUBRIC_NARRATIVITY, 5)
    }
    
    if evaluation_task not in task_map:
        raise ValueError(f"Unknown evaluation task: {evaluation_task}")
        
    criteria, rubric, max_score = task_map[evaluation_task]
        
    # 3. Assemble and format the final prompt
    return SELENE_PROMPT.format(
        input=input_str,
        output=output_str,
        criteria=criteria,
        rubric=rubric,
        max_score=max_score
    )