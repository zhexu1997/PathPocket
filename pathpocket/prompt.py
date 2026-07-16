from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# Custom entity extraction prompts for medical/pathology domain
PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are an Expert Pathology Knowledge Graph Specialist. Your goal is to extract structured medical entities and relationships from texts. The knowledge must be accurate, universal, and useful for patients, students, and clinicians.

---Instructions---
1.  **Entity Extraction:**
    *   **Identification:** Identify clinically and pathologically significant entities.
    *   **Entity Details:**
        *   `entity_name`: Use standard medical terminology. Capitalize Title Case. Ensure consistency (e.g., use "Renal Cell Carcinoma" consistently).
        *   `entity_type`: Categorize using: `{entity_types}`. If none apply, use `Other`.
        *   `entity_description`: Provide a concise definition focusing on clinical or pathological significance (e.g., etiology, morphology, mechanism). Avoid context-specific temporality (e.g., "observed in this patient").
    *   **Output Format:** `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction:**
    *   **Identification:** Identify objective, factual connections such as etiology, pathogenesis, diagnostic criteria, treatment efficacy, or prognosis.
    *   **Multi-Entity Relations:** Capture relationships involving two or more entities (e.g., "Drug X combined with Drug Y treats Disease Z in Organ W" should be a single relation connecting all four entities).
    *   **Relationship Details:**
        *   `entities`: List of ALL entity names involved in this relationship, separated by `{tuple_delimiter}`. Must match extracted entity names exactly.
        *   `relationship_keywords`: High-level medical concepts. Comma-separated.
        *   `relationship_description`: Explain the medical logic connecting ALL the entities (e.g., mechanism of action, causal link, synergistic effects).
    *   **Output Format:** `relation{tuple_delimiter}entity_name_1{tuple_delimiter}entity_name_2{tuple_delimiter}...{tuple_delimiter}entity_name_n{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`
    *   **Note:** The last two fields are always keywords and description.

3.  **General Protocols:**
    *   **Delimiter:** Use `{tuple_delimiter}` strictly as a separator.
    *   **Directionality:** Relationships like "treats," "causes," or "indicates" are directed. Ensure logical flow.
    *   **Objectivity:** Use third-person medical language. No pronouns.
    *   **Language:** Output in {language}. Keep proper nouns (e.g., gene names like *EGFR*, drugs) in standard medical English.
    *   **Completion:** End with `{completion_delimiter}`.

---Examples---
{examples}

---Input Data---
Entity_types: [{entity_types}]
Text:
{input_text}

"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract pathology-related entities and relationships from the input text.

---Instructions---
1.  **Format:** Strictly follow the system prompt's delimiter and field requirements.
2.  **Content:** Output *only* the list of entities and relationships.
3.  **Medical Accuracy:** Ensure entities represent generalizable medical knowledge, not specific patient case data unless it illustrates a general principle.
4.  **Completion:** End with `{completion_delimiter}`.
5.  **Language:** {language}.

<Output>
"""

# PROMPTS["entity_extraction_examples"] = [
#     """<Input Text>
# Coagulative necrosis is the most common pattern of cell death, primarily caused by ischemia in solid organs such as the heart and kidney. Grossly, the affected tissue appears pale and firm. Microscopically, the basic cell outline is preserved for several days despite the loss of the nucleus, a process resulting from the denaturation of structural proteins.

# <Output>
# entity{tuple_delimiter}Coagulative Necrosis{tuple_delimiter}PathologicalFinding{tuple_delimiter}A form of cell death characterized by the preservation of tissue architecture for a limited timespan.
# entity{tuple_delimiter}Ischemia{tuple_delimiter}Pathogenesis{tuple_delimiter}A restriction in blood supply to tissues, causing a shortage of oxygen that represents the primary cause of coagulative necrosis.
# entity{tuple_delimiter}Heart{tuple_delimiter}AnatomicalSite{tuple_delimiter}A solid organ susceptible to ischemic injury and coagulative necrosis (myocardial infarction).
# entity{tuple_delimiter}Kidney{tuple_delimiter}AnatomicalSite{tuple_delimiter}A solid organ commonly affected by ischemic coagulative necrosis (renal infarction).
# entity{tuple_delimiter}Denaturation of Proteins{tuple_delimiter}Pathogenesis{tuple_delimiter}The biochemical process where structural proteins lose their shape, blocking proteolysis and preserving cell architecture.
# relation{tuple_delimiter}Ischemia{tuple_delimiter}Coagulative Necrosis{tuple_delimiter}etiology, cause{tuple_delimiter}Ischemia is the primary causative factor leading to coagulative necrosis in solid organs.
# relation{tuple_delimiter}Ischemia{tuple_delimiter}Coagulative Necrosis{tuple_delimiter}Heart{tuple_delimiter}Kidney{tuple_delimiter}location, susceptibility, organ involvement{tuple_delimiter}Ischemia causes coagulative necrosis in solid organs, particularly affecting the heart (myocardial infarction) and kidney (renal infarction).
# relation{tuple_delimiter}Denaturation of Proteins{tuple_delimiter}Coagulative Necrosis{tuple_delimiter}mechanism, pathogenesis{tuple_delimiter}Denaturation of structural proteins prevents immediate enzymatic digestion, leading to the preserved architecture seen in coagulative necrosis.
# {completion_delimiter}

# """,
#     """<Input Text>
# Invasive Ductal Carcinoma (IDC) is the most common type of breast cancer. Immunohistochemistry is vital for treatment planning. Overexpression of the HER2 protein indicates an aggressive tumor phenotype but predicts a favorable response to targeted therapies like Trastuzumab. Conversely, tumors expressing Estrogen Receptor (ER) are candidates for hormonal therapy with Tamoxifen.

# <Output>
# entity{tuple_delimiter}Invasive Ductal Carcinoma{tuple_delimiter}Disease{tuple_delimiter}The most common histological type of breast cancer, arising from the ductal epithelium.
# entity{tuple_delimiter}HER2{tuple_delimiter}Biomarker{tuple_delimiter}A protein that, when overexpressed, promotes rapid cancer cell growth and serves as a therapeutic target.
# entity{tuple_delimiter}Trastuzumab{tuple_delimiter}Drug{tuple_delimiter}A monoclonal antibody drug used to target HER2-positive cancer cells.
# entity{tuple_delimiter}Estrogen Receptor{tuple_delimiter}Biomarker{tuple_delimiter}A nuclear receptor that binds estrogen; its presence indicates potential responsiveness to hormonal therapy.
# entity{tuple_delimiter}Tamoxifen{tuple_delimiter}Drug{tuple_delimiter}A selective estrogen receptor modulator (SERM) used to treat ER-positive breast cancer.
# relation{tuple_delimiter}HER2{tuple_delimiter}Invasive Ductal Carcinoma{tuple_delimiter}prognosis, stratification{tuple_delimiter}Overexpression of HER2 in IDC indicates a more aggressive disease course.
# relation{tuple_delimiter}HER2{tuple_delimiter}Invasive Ductal Carcinoma{tuple_delimiter}Trastuzumab{tuple_delimiter}predictive marker, drug target, treatment{tuple_delimiter}HER2 overexpression in IDC predicts clinical response to the targeted therapy Trastuzumab.
# relation{tuple_delimiter}Estrogen Receptor{tuple_delimiter}Invasive Ductal Carcinoma{tuple_delimiter}Tamoxifen{tuple_delimiter}predictive marker, drug target, hormonal therapy{tuple_delimiter}Expression of ER in IDC indicates that the tumor is likely to respond to hormonal therapy with Tamoxifen.
# {completion_delimiter}

# """,
# ]

PROMPTS["entity_extraction_examples"] = [
    """<Input Text>
Lynch Syndrome (Hereditary Non-Polyposis Colorectal Cancer) is an autosomal dominant condition caused by germline mutations in DNA mismatch repair (MMR) genes, most commonly MLH1 and MSH2. Defects in these genes lead to Microsatellite Instability (MSI), a molecular phenotype characterized by hypermutability. Colorectal carcinomas exhibiting MSI-High status are distinct; they often have a mucinous histology and are typically resistant to 5-Fluorouracil (5-FU) based adjuvant chemotherapy. However, MSI-High tumors are highly responsive to immune checkpoint inhibitors like Pembrolizumab.

<Output>
entity{tuple_delimiter}Lynch Syndrome{tuple_delimiter}Disease{tuple_delimiter}An inherited disorder increasing the risk of many types of cancer, primarily colorectal cancer.
entity{tuple_delimiter}MLH1{tuple_delimiter}Gene{tuple_delimiter}A DNA mismatch repair gene frequently mutated in Lynch Syndrome.
entity{tuple_delimiter}MSH2{tuple_delimiter}Gene{tuple_delimiter}A major DNA mismatch repair gene associated with hereditary non-polyposis colorectal cancer.
entity{tuple_delimiter}Microsatellite Instability{tuple_delimiter}MolecularPathway{tuple_delimiter}A condition of genetic hypermutability resulting from impaired DNA mismatch repair.
entity{tuple_delimiter}5-Fluorouracil{tuple_delimiter}Drug{tuple_delimiter}An antimetabolite chemotherapy medication used to treat cancer.
entity{tuple_delimiter}Pembrolizumab{tuple_delimiter}Drug{tuple_delimiter}A monoclonal antibody that binds to the PD-1 receptor, used in immunotherapy.
relation{tuple_delimiter}Lynch Syndrome{tuple_delimiter}MLH1{tuple_delimiter}MSH2{tuple_delimiter}etiology, genetic cause{tuple_delimiter}Lynch Syndrome is caused by germline mutations in mismatch repair genes, specifically MLH1 and MSH2.
relation{tuple_delimiter}MLH1{tuple_delimiter}MSH2{tuple_delimiter}Microsatellite Instability{tuple_delimiter}mechanism, molecular consequence{tuple_delimiter}Defects in the MLH1 or MSH2 genes result in the failure of DNA repair, leading to Microsatellite Instability.
relation{tuple_delimiter}Microsatellite Instability{tuple_delimiter}5-Fluorouracil{tuple_delimiter}Pembrolizumab{tuple_delimiter}predictive marker, drug response{tuple_delimiter}Tumors with Microsatellite Instability are resistant to 5-Fluorouracil but show favorable response to Pembrolizumab.
{completion_delimiter}

""",
    """<Input Text>
Alzheimer's Disease is the most common cause of dementia. Neuropathological diagnosis requires the observation of two key findings: extracellular Amyloid Plaques and intracellular Neurofibrillary Tangles. The tangles are composed of hyperphosphorylated Tau protein, which aggregates and disrupts microtubule stability. These pathological changes predominantly begin in the Entorhinal Cortex and Hippocampus before spreading to the neocortex. The presence of the APOE e4 allele is a major genetic risk factor that promotes amyloid accumulation.

<Output>
entity{tuple_delimiter}Alzheimer's Disease{tuple_delimiter}Disease{tuple_delimiter}A progressive neurodegenerative disorder and the leading cause of dementia.
entity{tuple_delimiter}Amyloid Plaques{tuple_delimiter}PathologicalFinding{tuple_delimiter}Extracellular deposits of beta-amyloid protein found in the brains of Alzheimer's patients.
entity{tuple_delimiter}Neurofibrillary Tangles{tuple_delimiter}PathologicalFinding{tuple_delimiter}Intracellular aggregates of hyperphosphorylated tau protein.
entity{tuple_delimiter}Tau protein{tuple_delimiter}Biomarker{tuple_delimiter}A protein that stabilizes microtubules; its hyperphosphorylation leads to tangle formation.
entity{tuple_delimiter}Entorhinal Cortex{tuple_delimiter}AnatomicalSite{tuple_delimiter}An area of the brain located in the medial temporal lobe, an early site of Alzheimer's pathology.
entity{tuple_delimiter}Hippocampus{tuple_delimiter}AnatomicalSite{tuple_delimiter}A complex brain structure embedded deep into the temporal lobe, critical for learning and memory.
entity{tuple_delimiter}APOE e4{tuple_delimiter}Gene{tuple_delimiter}A variant of the Apolipoprotein E gene implicated as a strong risk factor for Alzheimer's.
relation{tuple_delimiter}Alzheimer's Disease{tuple_delimiter}Amyloid Plaques{tuple_delimiter}Neurofibrillary Tangles{tuple_delimiter}diagnostic criteria, pathological hallmark{tuple_delimiter}The definitive neuropathological diagnosis of Alzheimer's Disease relies on the presence of Amyloid Plaques and Neurofibrillary Tangles.
relation{tuple_delimiter}Neurofibrillary Tangles{tuple_delimiter}Tau protein{tuple_delimiter}composition, molecular pathogenesis{tuple_delimiter}Neurofibrillary Tangles are physically composed of aggregated, hyperphosphorylated Tau protein.
relation{tuple_delimiter}Alzheimer's Disease{tuple_delimiter}Entorhinal Cortex{tuple_delimiter}Hippocampus{tuple_delimiter}location, disease progression{tuple_delimiter}Pathology in Alzheimer's Disease typically initiates in the Entorhinal Cortex and Hippocampus.
relation{tuple_delimiter}APOE e4{tuple_delimiter}Alzheimer's Disease{tuple_delimiter}Amyloid Plaques{tuple_delimiter}risk factor, genetic association{tuple_delimiter}The APOE e4 allele increases the risk of Alzheimer's Disease by promoting the accumulation of Amyloid Plaques.
{completion_delimiter}

"""
]

# Continue extraction prompt for gleaning
PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relations:** Output at least 5 fields for each relation, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`. The last two fields are always keywords and description. All fields in between are entity names (2 or more entities).
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 10 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""


PROMPTS["rag_response_mm"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph, Document Chunks and Retrieved Similar Images found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data`, `Document Chunks`, and `Retrieved Similar Images` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - **Image handling (critical)**: Do NOT rely on your own visual interpretation of any images. Treat the `Retrieved Similar Images` section as an external knowledge source rather than as raw pixels to interpret. 
  - Consider the evidence source, evidence level, image similarity (if any), and anatomical structure match degree. Higher evidence level (smaller number), higher image similarity, and higher anatomical match degree indicate more trustworthy evidence. Prioritize information from more credible images when synthesizing the answer.
  - When there is any ambiguity in images (e.g., patterns that could be read multiple ways), prefer the **retrieval-grounded descriptions** in `Retrieved Similar Images` and `Document Chunks` over any intuition. 
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - When multiple sources conflict, rely on those with higher evidence level (smaller number), higher image similarity, and higher anatomical structure match degree.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 10 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response_mm"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks and Retrieved Similar Images found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Document Chunks` and `Retrieved Similar Images` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - **Image handling (critical)**: Do NOT rely on your own visual interpretation of any images. Treat the `Retrieved Similar Images` section as an external knowledge source rather than as raw pixels to interpret.
  - Consider image similarity (if any) when weighing evidence. Higher image similarity indicates more trustworthy similar-case descriptions.
  - When there is any ambiguity, prefer **retrieval-grounded descriptions** in `Retrieved Similar Images` and `Document Chunks` over any intuition.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - When multiple sources conflict, prefer higher image similarity and stronger textual support from document chunks.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 10 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""



PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by intelligently combining information from the provided **Context** and your internal knowledge.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must prioritize relevant facts from the Document Chunks found in the **Context**, and supplement with your internal knowledge only when necessary.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Relevance Assessment & Knowledge Selection:
  - **First, assess the relevance** of each Document Chunk in the **Context** to the user query. Identify which chunks are directly relevant, partially relevant, or irrelevant.
  - **Prioritize Context Knowledge**: Always prioritize using information from the **Context** that is relevant to the query. The context knowledge should form the primary basis of your answer.
  - **Use Internal Knowledge as Fallback**: Only use your internal knowledge when:
    a) The context does not contain sufficient information to answer the query, OR
    b) The context information is incomplete or unclear, OR
    c) You need to provide general background information to help interpret the context (but clearly distinguish this from context-based facts)
  - **Clearly indicate the source**: When using context knowledge, cite the references. When using internal knowledge, acknowledge it appropriately.

2. Step-by-Step Answer Generation:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query, considering their relevance scores.
  - **Primary Answer from Context**: Construct the main answer using relevant facts from the Document Chunks. Weave the extracted facts into a coherent and logical response.
  - **Supplement with Internal Knowledge**: If the context is insufficient, supplement with your internal knowledge to provide a complete answer. However, clearly distinguish between context-based facts and general knowledge.
  - Track the reference_id of the document chunks which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the context-based facts presented in the response.
  - Do not generate anything after the reference section.

3. Content & Grounding:
  - **Context Priority**: When context information is available and relevant, it takes precedence over internal knowledge. Do not contradict context facts with internal knowledge.
  - **Internal Knowledge Usage**: Use internal knowledge to:
    * Fill gaps when context is incomplete
    * Provide general background or definitions
    * Connect context facts with broader medical/scientific understanding
  - **Transparency**: Be transparent about what comes from context vs. internal knowledge when the distinction is important.

4. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

5. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 10 most relevant citations.
  - Only include references that directly support context-based facts in your answer.
  - Use the exact `reference_id` from the Reference Document List in brackets (e.g. `[3]` if that chunk's reference_id is 3). The References section lists only sources you cited; numbering may be non-consecutive when you skip unused retrieval IDs.
  - In the References section, use the **Document Title** from the Reference Document List (not file paths or technical filenames like `raw.nxml`).
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

6. Reference Section Example:
```
### References

- [1] Document Title One
- [3] Document Title Three
```

7. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""


# Pathology query → structured retrieval object (normalized downstream to core/candidate entities & relationships)
PROMPTS["keywords_extraction"] = """---Role---
You are an expert pathology query analyzer for a multimodal pathology knowledge hypergraph RAG system. Decompose the user question into a **single JSON object** for retrieval. Output language for string values: {language}. Keep standard drug/gene symbols in common medical English when natural.

---Goal---
Return one JSON object with **exactly** these keys:

- **site**: **One string only** — anatomical site, organ, specimen source, or localization. Use `""` if none.
- **gross_entities**: Macroscopic / specimen-level cue phrases — JSON **array of strings** or one string (e.g. specimen color, size, number of pieces, cut surface). Use `[]` if there is no gross/specimen narrative.
- **gross_description**: **One string only** — copy or lightly trim the **original gross / specimen / 送检** description from the stem. 
- **morphology_entities**: Named **microscopic** morphology / histology entities — JSON **array of strings** or one string.
- **morphology_description**: **One string only** — copy or lightly trim the **original microscopic / histology** description from the stem. Keep the question’s wording; do not replace with abstract keywords. Use `""` if there is no microscopic text.
- **marker_entities**: IHC / molecular marker names — array of strings or one string.
- **marker_description**: **One string only** — copy the **original** immunohistochemistry / special stain / molecular lines from the stem, verbatim or with minimal cleanup. Use `""` if none.
- **clinical_entities**: Clinical / setting entities (besides age/sex) — array of strings or one string.
- **clinical_description**: **One string only** — start with **sex and age** exactly as in the stem. Use `""` only if the query truly has no such info. 
- **other_entities**: Other key entities/concepts from the stem that don't fit the above categories — array of strings or one string.
- **candidate_answer**: MCQ: **list** of option texts (one string per option). Otherwise `[]` or one short string.

---Instructions---
1. **Output**: Valid JSON only — no markdown fences, no text before/after.
2. **Source split (CRITICAL)**:
   - **Stem-only**: You MUST derive **site**, **gross_entities**, **gross_description**, **morphology_entities**, **morphology_description**, **marker_entities**, **marker_description**, **clinical_entities**, **clinical_description**, **other_entities** ONLY from the **stem**. Do NOT use the MCQ options to invent or supplement these fields.
   - **Options-only**: You MUST derive **candidate_answer** ONLY from the option texts when options are present.
   - If the query contains both stem and options, treat them as two separate sources with the above rules.
3. **Types**: `site`, `gross_description`, `morphology_description`, `marker_description`, and `clinical_description` must be **strings** (never JSON arrays). Use `""` when empty.
4. **Concise** for `site` and `*_entities`; description fields may stay close to full original sentences where helpful.
5. **Non-pathology / garbage queries**: Return all keys with `[]` or `""` as appropriate.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1: Bone marrow biopsy

Query:
女，69岁。送检：（髂后上棘）灰白色组织3条，长约0.5-1.2cm，直径均约03cm，全取1盒。
送检（髂后上棘）骨髓造血面积约60%，三系造血细胞可见，红系增生活跃，以中晚幼红为主，散在或成岛分布，部分细胞偏幼稚；粒系以中晚幼粒为主，见少量分叶核和杆状核；巨核以成熟巨核为主，约2-10个/HPF。
免疫组化（01#）：网状纤维染色（1+）。最可能的病理诊断是什么？

选项：
A. 缺铁性贫血
B. 骨髓增生活跃
C. 类白血病反应
D. 骨髓增生异常综合征
E. 急性白血病前期

请明确给出一个大写字母作为最后答案（A~E），并在答案后说明理由。

Output:
{
  "site": "髂后上棘、骨髓",
  "gross_entities": ["髂后上棘", "灰白色组织"],
  "gross_description": "（髂后上棘）灰白色组织3条，长约0.5-1.2cm，直径均约03cm，全取1盒。",
  "morphology_entities": ["中晚幼红细胞", "中晚幼粒细胞", "分叶核", "杆状核", "成熟巨核细胞"],
  "morphology_description": "骨髓造血面积约60%，三系造血细胞可见，红系增生活跃，以中晚幼红为主，散在或成岛分布，部分细胞偏幼稚；粒系以中晚幼粒为主，见少量分叶核和杆状核；巨核以成熟巨核为主，约2-10个/HPF。",
  "marker_entities": ["网状纤维染色"],
  "marker_description": "网状纤维染色（1+）。",
  "clinical_entities": [],
  "clinical_description": "女、69岁",
  "other_entities": [],
  "candidate_answer": ["缺铁性贫血", "骨髓增生活跃", "类白血病反应", "骨髓增生异常综合征", "急性白血病前期"]
}

""",
    """Example 2: Neck mass vascular lesion

Query:
男，42岁。送检：（颈部肿物）灰红灰褐色不规则组织1块，大小约2.5*1.0*1.0cm，沿最大面切开组织，切面灰白灰褐色实性质中，全取12盒。
送检（颈部肿物）组织中见多量大小不等的脉管并局灶出血，管壁形态不规则，大小不一，部分管壁增厚。最可能的病理诊断是什么？

选项：
A. 神经纤维瘤
B. 淋巴管瘤
C. 脉管瘤
D. 血管瘤
E. 血管球瘤

请明确给出一个大写字母作为最后答案（A~E），并在答案后说明理由。

Output:
{
  "site": "颈部肿物",
  "gross_entities": ["颈部肿物", "灰红灰褐色不规则组织"],
  "gross_description": "（颈部肿物）灰红灰褐色不规则组织1块，大小约2.5*1.0*1.0cm，沿最大面切开组织，切面灰白灰褐色实性质中，全取12盒。",
  "morphology_entities": ["脉管", "局灶出血", "管壁增厚"],
  "morphology_description": "组织中见多量大小不等的脉管并局灶出血，管壁形态不规则，大小不一，部分管壁增厚。",
  "marker_entities": [],
  "marker_description": "",
  "clinical_entities": [],
  "clinical_description": "男、42岁",
  "other_entities": [],
  "candidate_answer": ["神经纤维瘤", "淋巴管瘤", "脉管瘤", "血管瘤", "血管球瘤"]
}

""",
    """Example 3: Temporal lobe glial lesion with IHC
Query:
女，5岁。冰冻送检：（颞叶肿瘤）灰白灰红色组织1块，大小约1.7*1.0*0.5cm，全取1盒。冰改。另送：（颞叶肿瘤1）灰黄灰红色软组织1块，大小约2.0*1.6*0.4cm，全取1盒；（颞叶肿瘤2）灰白灰红色脑组织1块，大小约5.0*5.0*2.5cm，切面呈灰白灰红色，实性，质软，取9盒。共取10盒。
送检（颞叶肿瘤1、颞叶肿瘤2）胶质细胞轻度增生，异型性不明显，未见血管内皮增生及坏死形成，未见病理性核分裂象，间质血管充血。
免疫组化（08#）：GFAP（+）、Oligo-2（+）、Neu-N（残存神经元+）、CD34（血管+）、IDH1（-）、ATRX（+）、Ki-67（+，约1%）。最可能的病理诊断是什么？

选项：
A. 局灶脑皮质发育不良
B. 神经节细胞瘤
C. 海马硬化
D. 低级别胶质瘤
E. 错构瘤

请明确给出一个大写字母作为最后答案（A~E），并在答案后说明理由。

Output:
{
  "site": "颞叶、脑",
  "gross_entities": ["颞叶肿瘤", "灰白灰红色组织", "灰黄灰红色软组织", "灰白灰红色脑组织"],
  "gross_description": "（颞叶肿瘤）灰白灰红色组织1块，大小约1.7*1.0*0.5cm，全取1盒。冰改。（颞叶肿瘤1）灰黄灰红色软组织1块，大小约2.0*1.6*0.4cm，全取1盒；（颞叶肿瘤2）灰白灰红色脑组织1块，大小约5.0*5.0*2.5cm，切面呈灰白灰红色，实性，质软，取9盒。共取10盒。",
  "morphology_entities": ["胶质细胞轻度增生", "间质血管充血"],
  "morphology_description": "（颞叶肿瘤1、颞叶肿瘤2）胶质细胞轻度增生，异型性不明显，未见血管内皮增生及坏死形成，未见病理性核分裂象，间质血管充血。",
  "marker_entities": ["GFAP", "Oligo-2", "Neu-N", "CD34", "IDH1", "ATRX", "Ki-67"],
  "marker_description": "免疫组化（08#）：GFAP（+）、Oligo-2（+）、Neu-N（残存神经元+）、CD34（血管+）、IDH1（-）、ATRX（+）、Ki-67（+，约1%）。",
  "clinical_entities": [],
  "clinical_description": "女、5岁",
  "other_entities": [],
  "candidate_answer": ["局灶脑皮质发育不良", "神经节细胞瘤", "海马硬化", "低级别胶质瘤", "错构瘤"]
}

""",
    """Example 4: Rectal tubular gland epithelium

Query:
男，55岁。送检：（直肠）灰白色组织5块，直径约0.3-1.1cm，全取3盒。
送检（直肠）组织由排列规则的管状腺体组成，部分杯状细胞减少或消失，腺体被覆单层或假复层柱状上皮，细胞排列紧密，核呈杆状、浓染，向肠腔上移，但总体不超过1/3，间质疏松水肿伴慢性炎细胞浸润。最可能的病理诊断是什么？

选项：
A. 直肠淋巴瘤病
B. 家族性腺瘤性息肉病
C. 直肠多发腺癌
D. 直肠多发类癌
E. 多发管状腺瘤

请明确给出一个大写字母作为最后答案（A~E），并在答案后说明理由。

Output:
{
  "site": "直肠",
  "gross_entities": ["直肠", "灰白色组织"],
  "gross_description": "（直肠）灰白色组织5块，直径约0.3-1.1cm，全取3盒。",
  "morphology_entities": ["管状腺体", "杯状细胞", "柱状上皮", "慢性炎细胞浸润"],
  "morphology_description": "（直肠）组织由排列规则的管状腺体组成，部分杯状细胞减少或消失，腺体被覆单层或假复层柱状上皮，细胞排列紧密，核呈杆状、浓染，向肠腔上移，但总体不超过1/3，间质疏松水肿伴慢性炎细胞浸润。",
  "marker_entities": [],
  "marker_description": "",
  "clinical_entities": [],
  "clinical_description": "男、55岁",
  "other_entities": [],
  "candidate_answer": ["直肠淋巴瘤病", "家族性腺瘤性息肉病", "直肠多发腺癌", "直肠多发类癌", "多发管状腺瘤"]
}

""",
]
"""
Prompt templates for PathPocket multimodal content processing
Based on caption and text information only (no vision model)
"""

PROMPTS = {}

# System prompts for different analysis types
PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = (
    "You are an expert medical image analyst. Analyze images based on their captions, "
    "footnotes, and surrounding context. Provide detailed, accurate descriptions."
)

PROMPTS["TABLE_ANALYSIS_SYSTEM"] = (
    "You are an expert medical data analyst. Analyze tables based on their captions, "
    "body content, and surrounding context. Provide detailed table analysis with specific insights."
)

PROMPTS["GENERIC_ANALYSIS_SYSTEM"] = (
    "You are an expert content analyst specializing in {content_type} content."
)

# Image analysis prompt template (caption-based, no vision model)
PROMPTS["image_caption_prompt"] = """Based on the following image information, provide a detailed analysis in JSON format:

{{
    "detailed_description": "A comprehensive description of the image based on available information:
    - Analyze the image caption and footnotes to understand what the image shows
    - Describe the likely content, structure, and key elements based on the caption
    - Explain relationships between elements mentioned in the caption
    - Note any medical or technical details mentioned
    - Connect the image content to the surrounding context when provided
    - Always use specific medical terminology and names instead of pronouns",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary of the image content and its significance based on caption and context (max 100 words)"
    }}
}}

Image Information:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}
{context_section}

Focus on extracting meaningful medical entities and relationships from the caption and context information."""

# Image analysis prompt with context support
PROMPTS["image_caption_prompt_with_context"] = """Based on the following image information and surrounding context, provide a detailed analysis in JSON format:

{{
    "detailed_description": "A comprehensive description of the image based on available information:
    - Analyze the image caption and footnotes to understand what the image shows
    - Describe the likely content, structure, and key elements based on the caption
    - Explain relationships between elements mentioned in the caption and how they relate to the surrounding context
    - Note any medical or technical details mentioned
    - Connect the image content to the surrounding context
    - Reference connections to the surrounding content when relevant
    - Always use specific medical terminology and names instead of pronouns",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "image",
        "summary": "concise summary of the image content, its significance, and relationship to surrounding content (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Image Information:
- Image Path: {image_path}
- Captions: {captions}
- Footnotes: {footnotes}

Focus on extracting meaningful medical entities and relationships from the caption, context, and their connections."""

# Table analysis prompt template (caption-based)
PROMPTS["table_caption_prompt"] = """Based on the following table information, provide a detailed analysis in JSON format:

{{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization based on caption and body content
    - Column headers and their meanings
    - Key data points and patterns visible in the table body
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented
    - Medical or clinical implications
    Always use specific names and values instead of general references.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose and key findings (max 100 words)"
    }}
}}

Table Information:
- Image Path: {table_img_path}
- Caption: {table_caption}
- Body: {table_body}
- Footnotes: {table_footnote}
{context_section}

Focus on extracting meaningful medical entities and relationships from the table caption, body, and context."""

# Table analysis prompt with context support
PROMPTS["table_caption_prompt_with_context"] = """Based on the following table information and surrounding context, provide a detailed analysis in JSON format:

{{
    "detailed_description": "A comprehensive analysis of the table including:
    - Table structure and organization based on caption and body content
    - Column headers and their meanings
    - Key data points and patterns visible in the table body
    - Statistical insights and trends
    - Relationships between data elements
    - Significance of the data presented in relation to surrounding context
    - How the table supports or illustrates concepts from the surrounding content
    - Medical or clinical implications
    Always use specific names and values instead of general references.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "table",
        "summary": "concise summary of the table's purpose, key findings, and relationship to surrounding content (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Table Information:
- Image Path: {table_img_path}
- Caption: {table_caption}
- Body: {table_body}
- Footnotes: {table_footnote}

Focus on extracting meaningful medical entities and relationships from the table caption, body, context, and their connections."""

# Generic content analysis prompt template
PROMPTS["generic_prompt"] = """Based on the following {content_type} content, provide a detailed analysis in JSON format:

{{
    "detailed_description": "A comprehensive analysis of the content including:
    - Content structure and organization
    - Key information and elements
    - Relationships between components
    - Context and significance
    - Relevant details for knowledge retrieval
    Always use specific terminology appropriate for {content_type} content.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of the content's purpose and key points (max 100 words)"
    }}
}}

Content: {content}
{context_section}

Focus on extracting meaningful information that would be useful for knowledge retrieval."""

# Generic content analysis prompt with context support
PROMPTS["generic_prompt_with_context"] = """Based on the following {content_type} content and surrounding context, provide a detailed analysis in JSON format:

{{
    "detailed_description": "A comprehensive analysis of the content including:
    - Content structure and organization
    - Key information and elements
    - Relationships between components
    - Context and significance in relation to surrounding content
    - How this content connects to or supports the broader discussion
    - Relevant details for knowledge retrieval
    Always use specific terminology appropriate for {content_type} content.",
    "entity_info": {{
        "entity_name": "{entity_name}",
        "entity_type": "{content_type}",
        "summary": "concise summary of the content's purpose, key points, and relationship to surrounding context (max 100 words)"
    }}
}}

Context from surrounding content:
{context}

Content: {content}

Focus on extracting meaningful information that would be useful for knowledge retrieval and understanding the content's role in the broader context."""

# Modal chunk templates
PROMPTS["image_chunk"] = """
Image Content Analysis (Caption-based):
Image Path: {image_path}
Captions: {captions}
Footnotes: {footnotes}

Analysis: {enhanced_caption}"""

PROMPTS["table_chunk"] = """Table Analysis (Caption-based):
Image Path: {table_img_path}
Caption: {table_caption}
Structure: {table_body}
Footnotes: {table_footnote}

Analysis: {enhanced_caption}"""

PROMPTS["generic_chunk"] = """{content_type} Content Analysis:
Content: {content}

Analysis: {enhanced_caption}"""



# Continue extraction prompt for gleaning
PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

# Delimiter constants (matching core.py)
# Define directly to avoid circular import
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
