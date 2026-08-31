population:
	python scripts/build_population.py --config config/population.yaml

population-audit:
	python scripts/build_population.py --config config/population.yaml --audit-inputs-only

population-check:
	python scripts/build_population.py --config config/population.yaml --validate-only
	pytest -q tests/population

validate-personas:
	python scripts/validate_personas.py

validate-prompts:
	python scripts/render_prompt_validation.py

validation-audit:
	python scripts/audit_validation_split.py

calibrate-primary:
	python scripts/run_primary_calibration.py

freeze-method:
	python scripts/freeze_method_manifest.py

validation-report:
	python scripts/build_validation_report.py
