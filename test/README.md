# Local test harness

This folder lets you run the container on a single case. It intentionally contains **no patient data**: the HECKTOR imaging data is governed by the challenge data use agreement and cannot be redistributed.

## Expected layout

Provide your own case under `test/input/` in the Grand Challenge format:

    test/input/
      ehr.json                          clinical record (see ehr.example.json)
      images/
        ct/  <your_case>__CT.nii.gz      or a .mha file
        pet/ <your_case>__PT.nii.gz      or a .mha file

Outputs are written to `test/output/`:

    test/output/
      images/head-neck-tumor-segmentation/output.mha
      t-stage.json
      n-stage.json
      rfs.json

## Clinical record format

`ehr.json` uses these exact keys, with integer-coded values:

| Key | Meaning |
|---|---|
| `CenterID` | Acquisition center identifier |
| `Age` | Age in years |
| `Gender` | Coded (e.g. 0 / 1) |
| `Tobacco Consumption` | Coded (e.g. 0 = no, 1 = yes) |
| `Alcohol Consumption` | Coded (e.g. 0 = no, 1 = yes) |
| `Performance Status` | Coded performance status |
| `Treatment` | Coded treatment category |
| `HPV Status` | Coded (e.g. 0 = negative, 1 = positive) |

See `ehr.example.json` for a template. Missing fields are median-imputed at inference, matching how the models were trained.
