# Project Specification

## Problem

Support teams receive tickets with inconsistent labels and priorities. This slows triage and makes recurring issues harder to identify.

## Users

### Support agent
- Reviews incoming tickets
- Corrects predicted category and priority
- Searches similar resolved tickets
- Reviews and edits suggested responses

### Support manager
- Monitors ticket volume and response trends
- Reviews model performance
- Identifies frequent issue categories
- Audits human corrections

## MVP user stories

1. Submit a ticket and receive a category and priority suggestion.
2. See why the baseline made its suggestion.
3. Correct the suggestion.
4. View category and priority distributions.
5. Reproduce model training and evaluation.

## Non-goals for v0.1

- Autonomous customer communication
- Production-scale authentication
- Fine-tuning a large language model
- Claiming meaningful performance from the demo dataset
- Complex microservice architecture
