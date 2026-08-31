.PHONY: help check clean manifest zenodo_citation validate-personas validate-prompts validation-audit calibrate-primary freeze-method validation-report

help:
	@echo "make check                  validate this submission (files, metadata, data)"
	@echo "make clean                  clean the raw Tier-1 export in raw_data_deposit/ into predictions/"
	@echo "make clean INPUT=raw.csv    clean a specific raw export instead"
	@echo "make manifest               fingerprint predictions/ and record them in metadata.json"
	@echo "make zenodo_citation        (re)generate .zenodo.json from metadata.json (Zenodo deposit metadata)"
	@echo "make validate-personas      validate G/F persona panels and skeletons"
	@echo "make validate-prompts       render and audit G/F prompt protocols offline"
	@echo "make validation-audit       audit validation split and holdout integrity"
	@echo "make calibrate-primary      fit C on primary development effects only"
	@echo "make freeze-method          freeze F/G/C method before holdout opening"
	@echo "make validation-report      build validation report from available outputs"

check:
	Rscript scripts/check.R

clean:
	@if [ -n "$(INPUT)" ]; then Rscript scripts/clean.R "$(INPUT)"; else Rscript scripts/clean.R; fi

manifest:
	Rscript scripts/manifest.R

zenodo_citation:
	Rscript scripts/zenodo_citation.R

validate-personas:
	python pipeline/scripts/validate_personas.py

validate-prompts:
	python pipeline/scripts/render_prompt_validation.py

validation-audit:
	python pipeline/scripts/audit_validation_split.py

calibrate-primary:
	python pipeline/scripts/run_primary_calibration.py

freeze-method:
	python pipeline/scripts/freeze_method_manifest.py

validation-report:
	python pipeline/scripts/build_validation_report.py
