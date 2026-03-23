# V5 Training Data Audit — 2026-03-22

## Summary

Current SFT data (htb_sft_v5.jsonl) has significant quality issues that limit model capability.
DPO data is much cleaner but can't fully compensate.

## SFT Data: 16,609 examples

| Type | Count | % | Quality | Issue |
|------|-------|---|---------|-------|
| microchain_fast_local | 5,722 | 34.5% | Decent | 31% shallow reasoning |
| smart_mentor | 5,174 | 31.2% | Mediocre | Generic 5-word justifications |
| phase_classifier | 4,569 | 27.5% | Good | Clean labels, realistic boards |
| smart_mentor_walkthrough | 1,144 | 6.9% | BAD | 96% contaminated |

### Critical: Walkthrough Contamination

- **1,100 / 1,144 (96%)** walkthrough-mined examples have `selected_command` that ISN'T a valid template name
- These teach the model to output raw shell commands (`certutil`, `cat user.txt`, literal prose)
- Directly conflicts with 10,896 clean template-based examples
- Examples of bad commands found:
  - `certutil -encode 20221105213715_output.zip b64.txt`
  - `cat user.txt`
  - `, Analyse the packets and we will find the flag on` (literal prose)
  - `nmap --open -p0- -sS -n -Pn -vvv --min-rate 5000 delivery.ht` (raw nmap, not template)
  - `/home/flag18/flag18: invalid option -- 'e'` (error output parsed as command)

### Shallow Reasoning

- **3,686 / 12,040 (31%)** mentor/microchain examples have reasoning < 30 characters
- Examples: "Version scan including SMB", "SSH and HTTP open", "Identify JWT authentication"
- No strategic depth — model learns syntax, not thinking

## DPO Data: 4,000 preference pairs

- **Much cleaner**: only 36/4000 (0.9%) bad chosen commands
- Shallow chosen reasoning: 637/4000 (16%) — better but still mediocre
- DPO will partially compensate for SFT issues

## Recommendations for V4 Retrain

1. **Filter out all 1,100 contaminated walkthrough examples** (or regenerate with proper template mapping)
2. **Use 32B teacher model** to enhance reasoning for all 12K mentor/microchain examples
   - Expand 5-word reasoning to 2-3 sentence tactical analysis
   - Add paraphrasing diversity
   - Options: Qwen2.5-32B-Instruct, Mixtral-8x7B, or cloud 70B if budget allows
3. **Validate all `selected_command` fields** match available template list before training
4. **Minimum reasoning length**: enforce >= 50 chars in data pipeline
5. **Consider**: running teacher locally on vast.ai GPU (32B fits in 32GB with 4-bit quant)
