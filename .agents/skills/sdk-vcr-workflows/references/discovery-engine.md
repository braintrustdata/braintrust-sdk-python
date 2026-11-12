# Discovery Engine Recording

Discovery Engine tests use HTTP VCR for sync REST calls and the test-only
`integrations/google_discoveryengine/_test_grpc.py` helper for async gRPC calls. Both
recording formats live under `py/src/braintrust/integrations/google_discoveryengine/cassettes/<version>/`.

### Prerequisites

- Install `gcloud` and use a Google Cloud project with billing and the
  `discoveryengine.googleapis.com` API enabled.
- Authenticate with Application Default Credentials (ADC). The account needs
  permission to invoke Discovery Engine in the target project. A `GEMINI_API_KEY`
  alone does not authenticate these tests.
- For answer and conversation tests, create a search app with generative responses
  enabled and attach a populated datastore. Wait for indexing to finish before
  recording. Ranking and grounding-check tests do not require indexed documents.

```sh
gcloud config set project YOUR_PROJECT_ID
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

The recording fixture obtains an access token using
`gcloud auth application-default print-access-token`, outside the recorded call.
Playback uses anonymous credentials and needs no Google account.

Live recording requires explicit resource IDs through environment variables.
There are no private-project defaults in the test code. Tests use the `global`
location. Ranking and grounding checks require the project; answer and
conversation coverage also requires the app and datastore.

A datastore can be populated from Google's public Alphabet earnings-report
sample PDFs at `gs://cloud-samples-data/gen-app-builder/search/alphabet-investor-pdfs`.
Wait for indexing to finish before recording.

```sh
export BRAINTRUST_GOOGLE_DISCOVERYENGINE_PROJECT="your-project"
export BRAINTRUST_GOOGLE_DISCOVERYENGINE_APP="your-app-id"
export BRAINTRUST_GOOGLE_DISCOVERYENGINE_DATASTORE="your-datastore-id"
```

### Record and replay

From `py/`, select a focused scenario. `--vcr-record=all` enables recording for
both REST and gRPC; the gRPC helper otherwise requires an existing cassette.

```sh
# REST ranking, including manual/setup entry-point coverage.
mise exec -- nox -s 'test_google_discoveryengine(latest)' -- --vcr-record=all -k 'test_rank and not test_rank_output_limit'

# Async gRPC ranking.
mise exec -- nox -s 'test_google_discoveryengine(latest)' -- --vcr-record=all -k 'test_async_grpc and rank'

# Verify all recordings without network access to Google.
mise exec -- nox -R -s 'test_google_discoveryengine(latest)' -- --vcr-record=none
```

Playback derives resource paths from the checked-in REST cassettes and ignores
resource environment overrides. No access to the recorded project's resources is
needed. The auto-instrument subprocess test also reads its ranking resource from
the cassette. Keep the project's recordings consistent when recording against a
different project.

The gRPC helper temporarily replaces one callable on one client transport,
records real protobuf requests/responses and errors as JSON, then restores the
callable. Playback reconstructs provider response objects and checks requests;
it does not exercise networking, retries, or real cancellation behavior.
HTTP authorization headers are filtered by shared VCR configuration; gRPC
recordings omit credentials and transport metadata. Inspect new recordings before
checking them in.

`generate_grounded_content` and `stream_generate_grounded_content` are deferred:
the v1 endpoints returned method-not-found errors during initial recording.
Do not fabricate success recordings for those methods.
