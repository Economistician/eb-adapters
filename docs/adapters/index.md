# Adapters

Adapters are the bridge between **real-world data or models** and the canonical
contracts defined in the Electric Barometer ecosystem.

They exist to translate *domain-specific*, *vendor-specific*, or *pipeline-specific*
representations into stable, governance-aware contract surfaces that downstream
systems can rely on.

---

## What is an adapter?

An adapter is **pure translation logic**.

It typically performs some combination of:

- Column mapping and renaming
- Type coercion and normalization
- Governance gate interpretation (e.g. observability, structural impossibility)
- Semantic alignment to a contract’s expectations

Adapters **do not**:
- Implement business rules
- Contain forecasting logic
- Encode evaluation thresholds or policies
- Mutate contract semantics

Those concerns belong upstream (data prep) or downstream (evaluation, optimization).

---

## Adapter categories

This repository currently contains two broad adapter classes:

### 1. Contract adapters

Contract adapters transform raw or feature-engineered datasets into
`eb-contracts` artifacts such as:

- `PanelDemandV1`
- Forecast result panels
- Readiness or evaluation artifacts

These adapters are **domain-aware but contract-faithful**.

Example:
- A QSR intraday demand table → `PanelDemandV1`

Contract adapters live under:

```
contracts/
└── demand_panel/
    └── v1/
        └── <domain>/
```

Domains (e.g. `qsr`) are organizational, not restrictive. Users are free to
introduce additional domains as needed.

---

### 2. Model adapters

Model adapters provide a common interface over forecasting or statistical libraries
(e.g. CatBoost, LightGBM, Prophet).

They standardize:
- Fit / predict semantics
- Input/output shapes
- Optional metadata surfaces

Model adapters live under:

```
models/
```

They are intentionally orthogonal to contract adapters.

---

## Writing your own adapter

You are **not required** to use the adapters shipped in this repository.

If your data already conforms to a contract—or if you prefer to implement your own
translation layer—you can do so independently.

The only requirement is that the final output satisfies the target contract.

See:

👉 **Writing Your Own Adapter**

This guide walks through:
- Adapter responsibilities
- Common patterns
- Validation expectations
- Integration with `eb-contracts`

---

## Design philosophy

Adapters are intentionally:

- **Explicit** – no hidden inference or heuristics
- **Composable** – easy to layer or replace
- **Non-authoritative** – contracts define truth, adapters translate into it
- **Replaceable** – users can swap or bypass them entirely

This separation ensures that Electric Barometer remains:
- Contract-first
- Domain-flexible
- Vendor-agnostic
- Governance-aligned

---

## When to add a new adapter

Add a new adapter when:

- A data source cannot naturally conform to a contract
- A domain introduces consistent semantic differences
- Reuse would prevent duplicated mapping logic
- Validation failures are systematic, not accidental

Do **not** add adapters merely to enforce policy—that belongs elsewhere.

---

## Summary

Adapters are glue—not authority.

They allow Electric Barometer contracts to remain stable while the real world
remains messy.

If contracts are the language of the ecosystem, adapters are the translators.
