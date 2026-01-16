## Hallucination / Answer Quality Summary (LLM-as-judge)

- Input: `evaluation_results/hallucination_report_judged.json`

### Worst 3: Normal RAG faithfulness

- **score=31** — What is the thesis submission deadline?
  - explanation: The answer states that there is not enough information in the provided context to answer a question, which is a claim that cannot be verified from the context. The context provides detailed information about the Master's Thesis process, including registration phases and responsibilities, which suggests that there is sufficient information available. Therefore, the answer is not fully supported by the context.
- **score=61** — In the Computer Networks study programme (2025-2026), list the compulsory courses in the 1st semester (names + course codes).
  - explanation: The answer correctly identifies 'Specification and Verification' as a compulsory course in the 1st semester, but it inaccurately states the academic year as 2025-2026, which is not mentioned in the context. Therefore, while the course information is supported, the year is not, leading to a medium faithfulness rating.
- **score=61** — In the Data Science study programme (2025-2026), which compulsory courses are in the 2nd semester?
  - explanation: The answer correctly lists 'Current Trends in Data Science and Artificial Intelligence' and 'Algorithmic Foundations of Data Science' as compulsory courses in the 2nd semester, but it incorrectly includes 'Bioinformatics' as a compulsory course, which is not mentioned as such in the context.

### Worst 3: Agentic RAG faithfulness

- **score=91** — What is the exam period for the Internet of Things course?
  - explanation: The answer accurately states that the project for the Internet of Things course is submitted at the end of the last full week before the exam period, which is confirmed in the context. It also correctly notes that the specific dates for the exam period are not provided, aligning with the information in the context.
- **score=91** — In the Computer Networks study programme (2025-2026), list the compulsory courses in the 1st semester (names + course codes).
  - explanation: The answer correctly identifies 'Data Science and Ethics' as a compulsory course in the 1st semester of the Computer Networks study programme, which is supported by the context. However, it does not mention that this course is worth 3 ECTS-credits or the contact hours, which are also part of the context. Overall, the answer is mostly accurate and does not contain contradictions or unsupported claims.
- **score=91** — In the Data Science study programme (2025-2026), which compulsory courses are in the 2nd semester?
  - explanation: The answer accurately reflects the limitations of the provided context, indicating that there is insufficient information to answer a specific question. It does not contain unsupported claims or contradictions, and it aligns well with the context provided.

### Baseline hallucinations (from judge baseline-vs-normal) — top 3

- **2 hallucinated claims** — Who teaches Database Systems and in which semester is it offered?
  - The course is typically taught by faculty members from the Computer Science department.
  - The specific instructor may vary from year to year.
- **1 hallucinated claims** — Which specialization (Data Science, Software Engineering, or Computer Networks) includes 'Future Internet' as a compulsory course, and what is its course code?
  - Course code INFR-202 is not mentioned in the context provided.
- **1 hallucinated claims** — Which compulsory course in the Software Engineering study programme is taught by Hans Vangheluwe?
  - The course name 'Model-Driven Software Engineering' is incorrect.
