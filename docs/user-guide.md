[English](user-guide.md) | [Tiếng Việt](user-guide.vi.md)

# User guide

Start with the [quickstart](../README.md). This guide follows the conversation workspace at revision `02dd6f9`.

1. Register/sign in, then upload JPEG, PNG or WEBP, or select an available sample. Missing sample files can leave the list empty.
2. Read the analysis and describe the desired edit and what to preserve, such as the background or evening warmth. Answer clarification questions where requested.
3. Discuss the proposed treatment. Chat prepares/updates a proposal without rendering pixels.
4. Execute the displayed proposal. Wait for the synchronous request to finish and inspect recorded operations and output.
5. Compare original/output at equal zoom: noise, halos, clipped highlights, skin texture and changes outside the target region.
6. Give refinement feedback, inspect the updated proposal and execute when ready. Export the chosen result. Ratings are subjective feedback, not ground truth.

## Demo and troubleshooting

Try a permitted backlit portrait: “Brighten the face slightly and preserve the background.” Review the proposed region, execute and inspect face detail/background. Request a smaller correction if skin looks flat. Include failures or unchanged results when explaining limits.

Source labels distinguish rule-based, VLM and VLM fallback. Internal scores are estimates, not accuracy percentages. A key alone does not prove Gemini ran. Expired clarification threads may need clarification again after restart. A busy session must finish before incompatible actions. See [upload limits](api.md). Saved flow logs are not streaming progress.

## Current screens

![Login](assets/login.png)

![New workspace](assets/workspace.png)

![Completed session](assets/session.png)

Captured from the local production build on 2026-10-30. The synthetic figure was uploaded to smoke-test registration, diagnosis and execution of an empty plan. This demonstrates navigation, not restoration quality. The UI displays “Phác đồ: Gemini” on this empty-plan result even though the server is rule-based; that label alone is not evidence of a Gemini call. Full real-photo refinement/export walkthrough and Markdown/Mermaid renderer checks remain unverified. See [verification](documentation-verification.md).
