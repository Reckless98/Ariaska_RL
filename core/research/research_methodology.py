"""
Research Methodology Framework for ARIASKA_RL

Provides tools and templates for rigorous research methodology documentation,
experimental design, and reproducible research practices.
"""

import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

console = Console()

@dataclass
class ResearchQuestion:
    """Structured research question definition"""
    id: str
    question: str
    hypothesis: str
    variables: Dict[str, str]  # independent, dependent, control
    significance: str
    methodology: str

@dataclass 
class ExperimentalDesign:
    """Comprehensive experimental design specification"""
    design_type: str  # randomized, factorial, within-subject, etc.
    sample_size: int
    power_analysis: Dict[str, float]
    controls: List[str]
    randomization_method: str
    blinding: str
    duration: str

@dataclass
class ResearchPaper:
    """Research paper structure and content"""
    title: str
    authors: List[str]
    abstract: str
    introduction: str
    methodology: str
    results: str
    discussion: str
    conclusion: str
    references: List[str]
    appendices: Dict[str, str]

class ResearchMethodology:
    """
    Comprehensive research methodology framework for ARIASKA_RL.
    
    Features:
    - Research question formulation templates
    - Experimental design guidelines
    - Statistical analysis protocols
    - Research paper generation
    - Literature review tools
    - Reproducibility checklists
    """
    
    def __init__(self, project_dir: str = "research_methodology"):
        self.project_dir = project_dir
        self.templates_dir = os.path.join(project_dir, "templates")
        self.protocols_dir = os.path.join(project_dir, "protocols")
        self.papers_dir = os.path.join(project_dir, "papers")
        
        # Create directory structure
        for directory in [self.templates_dir, self.protocols_dir, self.papers_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self._create_default_templates()
        console.print(f"[green]✓[/green] ResearchMethodology initialized at {project_dir}")
    
    def _create_default_templates(self):
        """Create default research templates and protocols"""
        
        # Research question template
        rq_template = {
            "research_questions": [
                {
                    "id": "RQ1",
                    "question": "How does [intervention] affect [outcome] in [population/context]?",
                    "hypothesis": "We hypothesize that [intervention] will [predicted effect] because [theoretical reasoning]",
                    "variables": {
                        "independent": "[intervention variable]",
                        "dependent": "[outcome measure]",
                        "control": "[variables to control for]"
                    },
                    "significance": "This research addresses [gap] and has implications for [application area]",
                    "methodology": "We will use [experimental design] with [sample size] to test this hypothesis"
                }
            ]
        }
        
        with open(os.path.join(self.templates_dir, "research_questions_template.json"), 'w') as f:
            json.dump(rq_template, f, indent=2)
        
        # Experimental design template
        ed_template = {
            "experimental_designs": {
                "randomized_controlled": {
                    "description": "Participants randomly assigned to experimental and control conditions",
                    "advantages": ["High internal validity", "Causal inference possible"],
                    "requirements": ["Random assignment", "Control group", "Adequate sample size"],
                    "considerations": ["Ethical issues", "External validity", "Practical constraints"]
                },
                "factorial_design": {
                    "description": "Multiple independent variables tested simultaneously",
                    "advantages": ["Tests interactions", "Efficient", "Real-world complexity"],
                    "requirements": ["Orthogonal factors", "Balanced design", "Power analysis"],
                    "considerations": ["Complexity interpretation", "Sample size requirements"]
                }
            }
        }
        
        with open(os.path.join(self.templates_dir, "experimental_designs.json"), 'w') as f:
            json.dump(ed_template, f, indent=2)
        
        # Statistical analysis protocol
        stats_protocol = {
            "analysis_protocol": {
                "descriptive_statistics": [
                    "Calculate means, standard deviations, and confidence intervals",
                    "Check for outliers using IQR method",
                    "Assess normality using Shapiro-Wilk test",
                    "Create visualizations (histograms, boxplots)"
                ],
                "inferential_statistics": [
                    "Select appropriate test based on data type and distribution",
                    "Check statistical assumptions",
                    "Set alpha level (typically 0.05)",
                    "Calculate effect sizes",
                    "Interpret practical significance"
                ],
                "multiple_comparisons": [
                    "Apply Bonferroni correction if needed",
                    "Consider False Discovery Rate (FDR) control",
                    "Report adjusted p-values"
                ],
                "reporting": [
                    "Report test statistics, p-values, and effect sizes",
                    "Include confidence intervals",
                    "Provide practical interpretation",
                    "Discuss limitations"
                ]
            }
        }
        
        with open(os.path.join(self.protocols_dir, "statistical_analysis.json"), 'w') as f:
            json.dump(stats_protocol, f, indent=2)
    
    def create_research_proposal(self, title: str, research_questions: List[ResearchQuestion]) -> str:
        """Generate structured research proposal"""
        
        timestamp = datetime.now().isoformat()
        proposal = {
            "title": title,
            "generated": timestamp,
            "research_questions": [asdict(rq) for rq in research_questions],
            "sections": {
                "background": "TODO: Literature review and theoretical background",
                "objectives": "TODO: Primary and secondary research objectives", 
                "methodology": "TODO: Detailed experimental methodology",
                "expected_outcomes": "TODO: Anticipated results and impact",
                "timeline": "TODO: Project timeline and milestones",
                "resources": "TODO: Required resources and budget",
                "ethics": "TODO: Ethical considerations and approval"
            }
        }
        
        proposal_path = os.path.join(self.project_dir, f"research_proposal_{title.replace(' ', '_')}_{timestamp}.json")
        with open(proposal_path, 'w') as f:
            json.dump(proposal, f, indent=2)
        
        console.print(f"[cyan]📋[/cyan] Research proposal created: {proposal_path}")
        return proposal_path
    
    def design_experiment(self, research_question: ResearchQuestion, 
                         design_type: str = "randomized_controlled") -> ExperimentalDesign:
        """Design experiment based on research question"""
        
        # Load design templates
        with open(os.path.join(self.templates_dir, "experimental_designs.json"), 'r') as f:
            design_templates = json.load(f)
        
        if design_type not in design_templates["experimental_designs"]:
            console.print(f"[yellow]⚠[/yellow] Unknown design type: {design_type}")
            design_type = "randomized_controlled"
        
        # Create experimental design
        design = ExperimentalDesign(
            design_type=design_type,
            sample_size=self._calculate_sample_size(),
            power_analysis=self._perform_power_analysis(),
            controls=self._identify_controls(research_question),
            randomization_method="block_randomization",
            blinding="single_blind",
            duration="4_weeks"
        )
        
        return design
    
    def _calculate_sample_size(self, effect_size: float = 0.5, 
                              power: float = 0.8, alpha: float = 0.05) -> int:
        """Calculate required sample size for adequate statistical power"""
        
        # Simplified Cohen's d calculation for t-test
        # More sophisticated calculations would use specific statistical libraries
        
        # Basic approximation for two-group comparison
        if effect_size == 0.2:  # small effect
            return 80
        elif effect_size == 0.5:  # medium effect  
            return 32
        elif effect_size == 0.8:  # large effect
            return 20
        else:
            # Linear interpolation for other effect sizes
            if effect_size < 0.2:
                return min(100, int(80 / (effect_size / 0.2)))
            elif effect_size < 0.5:
                return int(80 - (effect_size - 0.2) * (80 - 32) / 0.3)
            elif effect_size < 0.8:
                return int(32 - (effect_size - 0.5) * (32 - 20) / 0.3)
            else:
                return max(15, int(20 / (effect_size / 0.8)))
    
    def _perform_power_analysis(self) -> Dict[str, float]:
        """Perform statistical power analysis"""
        return {
            "effect_size": 0.5,
            "alpha": 0.05,
            "power": 0.8,
            "required_n": 32,
            "actual_power": 0.8
        }
    
    def _identify_controls(self, research_question: ResearchQuestion) -> List[str]:
        """Identify control variables based on research question"""
        
        controls = []
        
        # Add standard controls for RL experiments
        if "learning" in research_question.question.lower():
            controls.extend(["random_seed", "initial_conditions", "environment_parameters"])
        
        if "agent" in research_question.question.lower():
            controls.extend(["agent_architecture", "hyperparameters", "training_duration"])
        
        if "performance" in research_question.question.lower():
            controls.extend(["evaluation_metrics", "test_conditions", "baseline_comparison"])
        
        # Add domain-specific controls
        controls.extend(["computational_resources", "implementation_version", "data_preprocessing"])
        
        return list(set(controls))  # Remove duplicates
    
    def generate_literature_review_template(self, topic: str) -> str:
        """Generate structured literature review template"""
        
        template = f"""# Literature Review: {topic}

## Search Strategy

### Databases Searched
- [ ] Google Scholar
- [ ] IEEE Xplore  
- [ ] ACM Digital Library
- [ ] ArXiv
- [ ] PubMed (if applicable)

### Search Terms
- Primary: {topic}
- Secondary: [related terms]
- Boolean combinations: [search strings]

### Inclusion Criteria
- [ ] Published in peer-reviewed journals/conferences
- [ ] Published within last 5 years (2019-2024)
- [ ] Written in English
- [ ] Relevant to research question

### Exclusion Criteria
- [ ] Not peer-reviewed
- [ ] Not available in full text
- [ ] Not relevant to research focus

## Literature Summary

### Key Themes
1. **Theme 1**: [Description]
   - Supporting studies: [citations]
   - Key findings: [summary]

2. **Theme 2**: [Description]
   - Supporting studies: [citations]
   - Key findings: [summary]

### Methodological Approaches
- **Approach 1**: [Description and frequency]
- **Approach 2**: [Description and frequency]

### Gap Analysis
- **Identified gaps**: [List gaps in current literature]
- **Research opportunities**: [How this work addresses gaps]

### Synthesis and Conclusions
[Overall synthesis of literature and implications for current research]

## References
[Formatted bibliography]
"""
        
        review_path = os.path.join(self.project_dir, f"literature_review_{topic.replace(' ', '_')}.md")
        with open(review_path, 'w') as f:
            f.write(template)
        
        console.print(f"[blue]📚[/blue] Literature review template created: {review_path}")
        return review_path
    
    def create_reproducibility_checklist(self) -> str:
        """Create reproducibility checklist for experiments"""
        
        checklist = """# Reproducibility Checklist

## Code and Implementation
- [ ] All code is version controlled (Git)
- [ ] Code is well-documented with comments
- [ ] Dependencies are specified with versions
- [ ] Installation instructions are provided
- [ ] Code style is consistent and follows guidelines

## Data Management
- [ ] Data sources are clearly documented
- [ ] Data preprocessing steps are recorded
- [ ] Raw data is preserved and accessible
- [ ] Data quality checks are implemented
- [ ] Missing data handling is documented

## Experimental Design
- [ ] Random seeds are set and documented
- [ ] Experimental parameters are recorded
- [ ] Control conditions are clearly defined
- [ ] Sample size justification is provided
- [ ] Power analysis is conducted

## Statistical Analysis
- [ ] Analysis plan is pre-specified
- [ ] Statistical assumptions are checked
- [ ] Multiple comparisons are addressed
- [ ] Effect sizes are reported
- [ ] Confidence intervals are provided

## Reporting
- [ ] Methods section is sufficiently detailed
- [ ] All results are reported (including null results)
- [ ] Limitations are discussed
- [ ] Code and data availability is stated
- [ ] Conflicts of interest are disclosed

## Verification
- [ ] Results can be reproduced by independent researcher
- [ ] Code runs without errors on different systems
- [ ] Documentation is sufficient for replication
- [ ] Key findings are robust to reasonable variations

---
*Use this checklist before submitting research for publication*
"""
        
        checklist_path = os.path.join(self.protocols_dir, "reproducibility_checklist.md")
        with open(checklist_path, 'w') as f:
            f.write(checklist)
        
        console.print(f"[green]✓[/green] Reproducibility checklist created: {checklist_path}")
        return checklist_path
    
    def generate_methods_section(self, experimental_design: ExperimentalDesign,
                                research_questions: List[ResearchQuestion]) -> str:
        """Generate methods section for research paper"""
        
        methods = f"""## Methods

### Experimental Design
This study employed a {experimental_design.design_type.replace('_', ' ')} design to investigate the research questions. The experiment was conducted over {experimental_design.duration} with {experimental_design.randomization_method.replace('_', ' ')} to ensure balanced assignment across conditions.

### Participants/Agents
The study included {experimental_design.sample_size} independent experimental runs. Sample size was determined through power analysis targeting 80% power to detect medium effect sizes (Cohen's d = 0.5) with α = 0.05.

### Procedure
1. **Initialization**: All experimental parameters were set according to the experimental design
2. **Randomization**: {experimental_design.randomization_method.replace('_', ' ')} was used to assign conditions
3. **Data Collection**: Measurements were collected at predetermined intervals
4. **Quality Control**: Data quality was monitored throughout the experiment

### Control Variables
The following variables were controlled across all experimental conditions:
"""
        
        for control in experimental_design.controls:
            methods += f"- {control.replace('_', ' ').title()}\n"
        
        methods += f"""
### Statistical Analysis
Statistical analyses were conducted following pre-registered protocols. Descriptive statistics were calculated for all variables. Inferential statistics appropriate to the data type and distribution were selected. Effect sizes and confidence intervals were calculated for all significant findings.

### Research Questions Addressed
"""
        
        for i, rq in enumerate(research_questions, 1):
            methods += f"{i}. {rq.question}\n"
        
        return methods
    
    def create_results_template(self, research_questions: List[ResearchQuestion]) -> str:
        """Create results section template"""
        
        results = """## Results

### Descriptive Statistics
[Table/summary of descriptive statistics for all variables]

### Primary Analyses
"""
        
        for i, rq in enumerate(research_questions, 1):
            results += f"""
#### Research Question {i}: {rq.question}

**Hypothesis**: {rq.hypothesis}

**Analysis**: [Statistical test used and justification]

**Results**: [Test statistic, p-value, effect size, confidence interval]

**Interpretation**: [Plain language interpretation of results]
"""
        
        results += """
### Secondary Analyses
[Any additional analyses, exploratory findings, or post-hoc tests]

### Robustness Checks
[Sensitivity analyses, alternative analysis approaches, assumption checks]
"""
        
        return results
    
    def generate_research_paper_template(self, title: str, authors: List[str],
                                       research_questions: List[ResearchQuestion],
                                       experimental_design: ExperimentalDesign) -> str:
        """Generate complete research paper template"""
        
        paper_template = f"""# {title}

**Authors**: {', '.join(authors)}

**Date**: {datetime.now().strftime('%B %d, %Y')}

## Abstract
[150-250 word summary covering background, methods, results, and conclusions]

## Introduction

### Background
[Literature review and theoretical background]

### Research Gap
[What gap in knowledge this research addresses]

### Research Questions and Hypotheses
"""
        
        for i, rq in enumerate(research_questions, 1):
            paper_template += f"""
**RQ{i}**: {rq.question}

**H{i}**: {rq.hypothesis}
"""
        
        paper_template += f"""
## Methods
{self.generate_methods_section(experimental_design, research_questions)}

{self.create_results_template(research_questions)}

## Discussion

### Summary of Findings
[Brief recap of main results]

### Theoretical Implications
[What do these results mean for theory?]

### Practical Implications
[What do these results mean for practice?]

### Limitations
[Study limitations and their implications]

### Future Research
[Directions for future investigation]

## Conclusion
[Brief conclusion summarizing contributions]

## References
[Formatted bibliography]

## Appendices

### Appendix A: Experimental Parameters
[Detailed experimental configuration]

### Appendix B: Additional Analyses
[Supplementary statistical analyses]

### Appendix C: Code Availability
[Information about accessing code and data]
"""
        
        paper_path = os.path.join(self.papers_dir, f"{title.replace(' ', '_')}_draft.md")
        with open(paper_path, 'w') as f:
            f.write(paper_template)
        
        console.print(f"[green]📄[/green] Research paper template created: {paper_path}")
        return paper_path
    
    def create_ethics_protocol(self) -> str:
        """Create research ethics protocol"""
        
        ethics = """# Research Ethics Protocol

## Ethical Considerations

### Human Subjects
- [ ] No human participants involved in this computational research
- [ ] If humans involved, IRB approval obtained
- [ ] Informed consent procedures documented
- [ ] Privacy and confidentiality protections in place

### Data Ethics
- [ ] Data sources are ethically obtained
- [ ] No proprietary or sensitive data used without permission
- [ ] Data sharing complies with applicable regulations
- [ ] Potential misuse of research findings considered

### AI Ethics
- [ ] AI system safety considerations addressed
- [ ] Potential for harmful applications assessed
- [ ] Bias and fairness implications considered
- [ ] Transparency and explainability requirements met

### Publication Ethics
- [ ] All contributors properly credited
- [ ] No plagiarism or self-plagiarism
- [ ] Conflicts of interest disclosed
- [ ] Data and code sharing plans specified

### Environmental Ethics
- [ ] Computational resource usage justified
- [ ] Energy consumption considered in experimental design
- [ ] Carbon footprint minimization strategies employed

## Risk Assessment
[Assessment of potential risks and mitigation strategies]

## Approval Status
- Ethics review required: [Yes/No]
- Review board: [Institution/Committee name]
- Approval date: [Date]
- Protocol number: [Number]
"""
        
        ethics_path = os.path.join(self.protocols_dir, "ethics_protocol.md")
        with open(ethics_path, 'w') as f:
            f.write(ethics)
        
        console.print(f"[green]🛡️[/green] Ethics protocol created: {ethics_path}")
        return ethics_path
    
    def display_research_guidelines(self):
        """Display research best practices and guidelines"""
        
        guidelines = """
🧪 **ARIASKA_RL Research Best Practices**

📋 **Planning Phase**
• Formulate clear, testable research questions
• Conduct thorough literature review
• Design robust experimental methodology
• Plan statistical analysis approach
• Consider ethical implications

🔬 **Execution Phase**  
• Follow pre-registered protocols
• Document all procedures and parameters
• Implement quality control measures
• Monitor data collection continuously
• Maintain detailed research logs

📊 **Analysis Phase**
• Follow pre-specified analysis plan
• Check statistical assumptions
• Report all results (including null findings)
• Calculate effect sizes and confidence intervals
• Perform robustness checks

📝 **Reporting Phase**
• Write clear, comprehensive methods
• Present results objectively
• Discuss limitations honestly
• Make code and data available
• Follow journal submission guidelines

♻️ **Reproducibility**
• Version control all code
• Document computational environment
• Set and record random seeds
• Provide installation instructions
• Enable independent verification
"""
        
        panel = Panel(guidelines, title="Research Methodology Guidelines", style="blue")
        console.print(panel)