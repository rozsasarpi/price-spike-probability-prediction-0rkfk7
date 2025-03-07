name: Feature request
description: Suggest an enhancement for the ERCOT RTLMP spike prediction system
title: "[FEATURE]: "
labels: ["enhancement"]
body:
  - type: markdown
    attributes:
      value: |
        ## Feature Request

        Thank you for suggesting an enhancement to the ERCOT RTLMP spike prediction system. Please fill out the following template to help us understand your request.
  - type: input
    attributes:
      label: Feature Name
      description: A concise name for the requested feature
      placeholder: e.g., Additional Price Threshold Support
    validations:
      required: true
  - type: textarea
    attributes:
      label: Problem Statement
      description: Describe the problem this feature would solve
      placeholder: When using the system, I find it difficult to...
    validations:
      required: true
  - type: textarea
    attributes:
      label: Proposed Solution
      description: Describe your proposed solution and how it addresses the problem
      placeholder: Adding support for...
    validations:
      required: true
  - type: dropdown
    attributes:
      label: Component
      description: Which component of the system would this feature affect?
      options:
        - Data Fetching
        - Feature Engineering
        - Model Training
        - Inference Engine
        - Backtesting Framework
        - Visualization & Metrics
        - CLI Interface
        - Documentation
        - Other (specify in description)
    validations:
      required: true
  - type: textarea
    attributes:
      label: Acceptance Criteria
      description: What specific criteria should be met for this feature to be considered complete?
      placeholder: |
        - The system should allow...
        - Performance should not degrade by more than...
        - Documentation should be updated to...
    validations:
      required: true
  - type: textarea
    attributes:
      label: Additional Context
      description: Any additional information, screenshots, or examples that help explain the feature request
    validations:
      required: false
  - type: checkboxes
    attributes:
      label: Checklist
      options:
        - label: I have checked that this feature request doesn't already exist
          required: true
        - label: I have considered how this feature aligns with the project's goals
          required: true
        - label: I have considered potential impacts on existing functionality
          required: true