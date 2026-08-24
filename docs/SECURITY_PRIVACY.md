# Security, privacy, and deployment controls

## Current prototype

The local Streamlit app processes uploaded bytes in the active process. It does not contain a database and does not intentionally write patient media or reports to disk. Video decoding uses a temporary file that is deleted after sampling. DICOM ingestion exposes only technical metadata in the interface.

This is not proof that an uploaded object is de-identified. Ultrasound images can contain burned-in patient names, medical record numbers, dates, facial profiles, or voice and screen-capture metadata. De-identification must occur before upload.

## Minimum controls before institutional research use

- approved REB or IRB protocol and data-management plan
- data-sharing agreements for multicenter transfer
- DICOM tag de-identification with a tested allowlist
- pixel-level burned-in text detection and manual verification
- encrypted storage and transport
- role-based access with least privilege
- institutional identity provider and multifactor authentication
- immutable audit logging
- separate development, validation, and production environments
- secrets in a managed secret store, never in GitHub or Streamlit configuration files
- dependency scanning, software bill of materials, signed releases, and patch process
- backup, retention, deletion, and incident-response procedures

## Deployment boundary

Do not expose this prototype on a public internet URL with clinical media. A public GitHub repository must contain code and synthetic test data only. Model weights can encode information about training data and require separate governance.

If the system is deployed inside a hospital network, use a reverse proxy with TLS, authentication, upload-size and file-type controls, malware scanning, request timeouts, and an isolated inference worker. Run the container as a non-root user with a read-only filesystem and no outbound network access unless a reviewed service requires it.

## External AI services

Do not send neonatal images to a general-purpose external vision or language service unless the institution has approved the service, contract, data location, retention policy, audit controls, and health-information terms. A service API key does not establish clinical validity or privacy compliance.

## Medical-device boundary

Software that interprets images for a medical purpose can be regulated as a medical device. In Canada, machine learning-enabled medical devices are subject to the Food and Drugs Act and Medical Devices Regulations. The intended use, risk classification, evidence, model change plan, transparency, monitoring, and quality system require specialist review.

See:

- [Health Canada pre-market guidance for machine learning-enabled medical devices](https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/application-information/guidance-documents/pre-market-guidance-machine-learning-enabled-medical-devices.html)
- [Good Machine Learning Practice guiding principles](https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/good-machine-learning-practice-medical-device-development.html)

