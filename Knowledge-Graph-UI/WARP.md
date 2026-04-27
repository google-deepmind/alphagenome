# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

This repository implements a Next.js 14 app-router UI for exploring Neo4j-based knowledge graphs in bioinformatics. The UI is schema-driven: a JSON "UI schema" describes data node/edge types, navigation, theming, and component wiring, and the frontend plus API routes render and query Neo4j accordingly. The same codebase is reused for multiple deployments (CFDE, Enrichr KG, lncRNAlyzr, Harmonizome, etc.) by changing configuration and environment.

Key references from the existing README:
- Run the dev server locally with `npm run dev` (or `yarn dev`) and open `http://localhost:3000`.
- Versioning for deployments is done with `npm version <major|minor|patch>`, which triggers Docker image build/push scripts and is used with Helm + `docker-compose config` for Kubernetes installation.

## Common commands

All commands assume you are in the repository root.

### Install & dev
- Install dependencies: `npm install`
- Run development server: `npm run dev`
  - Next.js dev server on port 3000; uses `NODE_ENV=development`.

### Build & run
- Production build: `npm run build`
  - Runs `next build` with `NODE_ENV=production`.
- Start production server from a standalone build: `npm run start`
  - Runs `node .next/standalone/server.js` (requires `npm run build` first).

### Linting
- Run Next.js/ESLint lint checks: `npm run lint`

### Versioning & Docker images
- Compose a version and update Docker image tags: `npm run compose`
  - Runs the TypeScript CLI `cli/version-compose.ts` with `ts-node` and environment from `.env` (via `dotenv`), updating `docker-compose` configuration.
- Create a new version tag and build the `kg` Docker image: `npm version <major|minor|patch>`
  - Script: `ts-node -r dotenv/config cli/version-compose.ts && docker-compose build kg`.
- Push the `kg` Docker image after versioning: `npm run postversion`
  - Script: `docker-compose push kg`.

### Kubernetes deployment (from README)
These commands rely on the `maayanlab/docker-compose` Helm chart and the repo's `docker-compose*.yml` files.
- Initial install:
  - `helm install <name> maayanlab/docker-compose -f <(docker-compose config) -n <name> --create-namespace`
- Upgrade existing release:
  - `helm upgrade <name> maayanlab/docker-compose -f <(docker-compose config) -n <name>`
- Render manifests locally:
  - `helm template <name> maayanlab/docker-compose -f <(docker-compose config) -n <name>`

### Revalidation and initialization helpers
These are HTTP endpoints used at runtime rather than CLI commands, but are important for debugging:
- Revalidate cached layout: `GET /api/revalidate` calls `revalidatePath('/', 'layout')`.
- Initialize color/edge metadata: `GET /api/initialize` precomputes edge/node color config and stores it in in-memory cache.
- Fetch UI schema: `GET /api/schema` returns the UI schema used by the app.

## Application architecture

### Next.js app router structure (`app/`)
- `app/layout.tsx`
  - Global root layout; imports `ThemeRegistry` and `fetch_kg_schema`.
  - `generateMetadata()` makes a network call to `fetch_kg_schema` and sets the page `<title>` and favicon from `schema.header.icon`.
  - `RootLayout` fetches the schema on the server and wraps the `children` in `ThemeRegistry`, passing `schema.ui_theme` (or falling back to `"cfde_theme"`).
- `app/ThemeRegistry.tsx`
  - Client component that wires Emotion + MUI + Next.js app router.
  - Exposes a `ThemeRegistry` component wrapped with `withCookie(ThemeRegistry)` from `components/ConsentCookie`.
  - Chooses the active MUI theme from the `themes/` modules (`cfde_theme`, `enrichr_kg_theme`, `lncRNAlyzr`, `harmonizome_kg_theme`) based on the `theme` prop set in `layout.tsx` from the UI schema.
  - Uses `useServerInsertedHTML` to integrate Emotion styles server-side, and conditionally renders Google Analytics (`nextjs-google-analytics`) depending on a consent cookie (`NEXT_PUBLIC_COOKIE_NAME`).
- `app/page.tsx` (root route `/`)
  - Server component that fetches the UI schema (`fetch_kg_schema`).
  - Selects the tab from `schema.header.tabs` whose `endpoint === '/'` and passes its `component` and `props` into the generic `Component` from `app/component_selector.tsx`.
  - Renders the header/subheader/footer components around the selected main component, controlled by `schema.header.fullWidth` and the `fullscreen` query parameter.
- `app/[page]/page.tsx` (dynamic routes like `/distillery`, `/download`, etc.)
  - Similar to `app/page.tsx`, but derives the endpoint from `params.page` and looks up matching `schema.header.tabs` entry whose `endpoint === '/${page}'`.
  - Uses the same layout pattern (Header, Subheader, Footer, QueryTranslator + Component).
- `app/component_selector.tsx`
  - Central dispatch point that maps schema `header.tabs[].component` strings (e.g. `"KnowledgeGraph"`, `"Enrichment"`, `"DistilleryLanding"`, `"Download"`, `"APIDoc"`, `"Tutorial"`, `"WholeNetwork"`, etc.) to actual React component implementations under `components/`.
  - Defines an `AsyncComponent` server component that calls the appropriate component (many are async server components) and wraps it in a `Suspense` boundary with a `CircularProgress` fallback.
  - This is where new top-level views should be wired when adding a new component type to the UI schema.
- `app/api/*`
  - Each subdirectory under `app/api` is a route group implementing the backend for graph queries, enrichment, documentation, and initialization (see next section).
  - `app/api` is also the folder scanned by `next-swagger-doc` to generate the OpenAPI spec.

### Core server-side utilities (`utils/`)
- `utils/neo4j.ts`
  - Creates a singleton `neo4jDriver` from `neo4j-driver`.
  - Chooses the Neo4j URL based on `NODE_ENV` and `NEO4J_VERSION`:
    - In development: `NEO4J_DEV_URL`.
    - In production: `NEO4J_V5_URL` if `NEO4J_VERSION === '5'`, else `NEO4J_URL`.
  - Auth is `neo4j.auth.basic(NEO4J_USER, NEO4J_PASSWORD)`.
- `utils/initialize.ts`
  - `fetch_kg_schema()`:
    - Loads a default schema from `public/schema.json` (typed as `UISchema` from `app/api/schema/route.ts`).
    - If `NEXT_PUBLIC_SCHEMA` is set, fetches the schema JSON from that URL instead (with ISR revalidation: `0` seconds in development, `3600` seconds otherwise).
  - `get_terms(node, search)`:
    - Utility for querying Neo4j to collect distinct property values across nodes, used mainly for precomputing value ranges in development or production.
  - `initialize_kg()`:
    - Consumes `UISchema` and returns:
      - `nodes`: keyed by node label, with schema metadata.
      - `edges`: sorted list of edge relation names based on `schema.edges[*].match`.
      - Tooltip templates for nodes and edges based on `display` arrays.
      - Default relations and gene-link relations as defined by the schema.
  - `initialize_enrichment()`:
    - Uses `schema.header.subheader` to build an `icon_picker` mapping that determines which libraries are associated with each UI icon/section.
- `utils/swagger.ts`
  - Uses `next-swagger-doc` to scan `app/api` and build an OpenAPI spec.
  - Cleans up empty HTTP method entries from the generated `paths` before returning the spec.
- `utils/client_side.ts`
  - Small client-only helpers (`'use client'`): query-string based `router_push` for app router, and a generic `usePrevious` hook.

### API routes (`app/api`)

These routes are central to how the UI talks to Neo4j and external services.

- `app/api/schema/route.ts`
  - Defines the `UISchema` TypeScript interface used throughout the app and in `utils/initialize.ts`.
  - `GET`: returns the current UI schema, caching it in `memory-cache` under the key `"schema"`.
- `app/api/initialize/route.ts` + `helper.ts`
  - `helper.ts::initialize()`:
    - Reads the UI schema and computes:
      - `colors` for node labels and edge relation types (based on `schema.nodes[*].color`, `schema.edges[*].color`, `order`, `ring_label`, `border_color`, etc.).
      - `aggr_scores`: min/max aggregates per configured numeric property, by running Cypher queries over Neo4j.
      - `edges`: list of relation names from `schema.edges[*].match`.
      - `arrow_shape` mapping from relation type to an arrow style (`ArrowShape` from `components/Cytoscape`).
  - `route.ts::GET`:
    - Caches the computed `{ aggr_scores, colors, edges, arrow_shape }` structure in `memory-cache` under `"initialize"` and returns it.
- `app/api/knowledge_graph/route.ts` (+ `helper.ts`, `resolver.ts`)
  - Implements the primary graph query endpoint `GET /api/knowledge_graph`.
  - Expects a `filter` query parameter containing JSON, validated with `zod` (`input_query_schema`). Parameters include `start`, `start_term`, optional `end`/`end_term`, `relation`, `limit`, `path_length`, `remove`, `expand`, `gene_links`, `augment`, etc.
  - Branches into three internal resolvers:
    - `resolve_two_terms`: shortest paths between two concrete nodes.
    - `resolve_term_and_end_type`: shortest paths between a concrete start node and any node of a given end type.
    - `resolve_one_term`: ego-network around a single term, with optional augmentation via co-expression (`augment_gene_set`).
  - Uses `initialize()` to get `edges`, `colors`, `aggr_scores`, and `arrow_shape` for proper coloring and line styling.
  - Internally uses `resolver.ts::resolve_results` to turn raw Neo4j paths into a normalized `NetworkSchema` (`nodes`/`edges` arrays) with colors and additional properties based on the schema and `utils/colors.ts`.
  - `helper.ts` defines `default_get_node_color_and_type`, `default_get_edge_color`, and constants like `highlight_color` used to emphasize search terms or score-based coloring.
- `app/api/enrichment/route.ts` (+ `helper.ts`)
  - `POST /api/enrichment` endpoint providing Enrichr-based enrichment analysis and returning a graph.
  - `helper.ts::enrichr_query()` fetches results from `NEXT_PUBLIC_ENRICHR_URL` (`/enrich?userListId=...&backgroundType=...`), applies regex-based label parsing (`get_regex`) per library, handles multi-direction/variant terms, tracks p-values/z-scores/log-p, and aggregates overlapping genes.
  - The route aggregates per-gene counts across libraries, filters by `min_lib`, `gene_degree`, `gene_limit`, etc., then maps Enrichr term labels into the UI schema: builds Cypher queries that connect those terms to gene nodes and optionally adds additional gene-gene edges ("gene_links") based on relations from `schema.edges`.
  - Uses `resolve_results` with a custom `get_node_color_and_type` that colors nodes based on enrichment p-values (`compute_colors`), and enriches nested `enrichment` arrays on each node.
- `app/api/docs/route.ts`
  - `GET /api/docs` returns the combined OpenAPI specification used by the `APIDoc` React component to render Swagger UI.
- `app/api/revalidate/route.ts`
  - Minimal route to trigger `revalidatePath('/', 'layout')` and return `{ revalidated: true, now: <timestamp> }`.
- Other routes:
  - `app/api/counter/route.ts`: simple counter endpoint useful for testing or demo purposes.
  - `app/api/initialize/*` and `app/api/enrichment/*` also contain helper modules used exclusively by the above routes.

### Components and visualization (`components/`)

- Top-level components are grouped by feature:
  - `TermAndGeneSearch/`: primary network search UI, including async form, network table, Cytoscape-based visualization, and tooltips.
  - `SimpleTermAndGeneSearch/`: simplified search interface.
  - `Enrichment/`: components backing the enrichment analysis view.
  - `Distillery/`: landing/use-case views for "Distillery"-style apps, including cards and use cases.
  - `Download/`: UI for downloading data and configuration (consumes markdown in `public/markdown` and related APIs).
  - `APIDoc/`: Swagger UI wrapper that calls `GET /api/docs`.
  - `WholeNetwork/`: components for rendering whole-network views.
  - `MarkdownComponent`, `SanitizedHTML`: content rendering utilities for markdown and sanitized HTML snippets driven by the schema.
  - Layout and chrome: `Header`, `Subheader`, `Footer`, `ConsentCookie`, `QueryTranslator`.
  - `Cytoscape/`: React Cytoscape.js integrations and styling (imported by network components and the API resolver for arrow shapes).
- Most high-level views are server components wired through `app/component_selector.tsx` and configured indirectly via `public/schema.json` or external schemas.

### Styling and theming

- MUI theming:
  - Theme modules in `themes/*.ts` define different brand themes; `ThemeRegistry` selects one based on `schema.ui_theme`.
  - Node/edge colors in graphs are further refined at runtime via `initialize()` and color utilities (`utils/colors.ts`).
- Global CSS and Tailwind:
  - `app/global.css` and `styles/*.css` provide baseline styles and knowledge-graph-specific tweaks (`kg.css`).
  - Tailwind is configured via `tailwind.config.ts` and `postcss.config.js`; Tailwind is mostly used for layout/spacing classes in the app router pages.

## Configuration & environment

Important environment variables used across the app (consult the code for full details):

### Neo4j connection
- `NEO4J_VERSION`: selects whether to use `NEO4J_V5_URL` vs `NEO4J_URL`.
- `NEO4J_URL`: base Neo4j URL for non-v5 production.
- `NEO4J_V5_URL`: base Neo4j URL for v5 deployments.
- `NEO4J_DEV_URL`: Neo4j URL in development (`NODE_ENV=development`).
- `NEO4J_USER`, `NEO4J_PASSWORD`: credentials for `neo4j-driver`.

### UI schema and host
- `NEXT_PUBLIC_SCHEMA`: if set, `utils/initialize::fetch_kg_schema()` fetches the UI schema JSON from this URL instead of using `public/schema.json`.
- `NEXT_PUBLIC_HOST`, `NEXT_PUBLIC_HOST_DEV`: base host used in enrichment queries and potentially other cross-service calls.
- `NEXT_PUBLIC_PREFIX`: optional path prefix; used in `next.config.mjs` as `basePath` and when constructing internal API URLs.

### Enrichr and analytics
- `NEXT_PUBLIC_ENRICHR_URL`: base URL for Enrichr API used by `app/api/enrichment/helper.ts`.
- `NEXT_PUBLIC_COOKIE_NAME`: cookie key used by `components/ConsentCookie` and `ThemeRegistry` to control whether Google Analytics is enabled.

### Misc
- Standard Next.js / Node environment variables like `NODE_ENV` control behavior of schema revalidation, logging, etc.

## Data, assets, and ingestion

- `public/schema.json` / `public/schema.example.json` / `public/reprotox_schema.json`:
  - Example and default UI schemas that demonstrate how to configure nodes, edges, header/footer layout, and tabs for different deployments.
  - Most UI customization is driven by these JSON files or by hosting a schema externally and referencing it via `NEXT_PUBLIC_SCHEMA`.
- `public/markdown/` and `public/tutorial/`:
  - Markdown and tutorial content displayed by `MarkdownComponent`, `Download`, and other content components referenced in the schema.
- Ingestion and Neo4j population scripts (`scripts/`):
  - `scripts/ingestion/`: Python-based tooling and Dockerfile to ingest CSV or other data sources into Neo4j (`import_csv.py`, `populate.py`, `rapid_populate.py`, `indexing.py`, `validation.py`, `ingest.sh`).
  - `scripts/neo4j/`: helper scripts and `requirements.txt` for managing Neo4j population outside the UI.
  - These are used to build and maintain the underlying graph database that the UI queries, but they are separate from the Next.js runtime.

## How to extend or debug

- To add a new top-level view or page:
  - Implement a React component (usually under `components/`) that accepts the props expected from the UI schema.
  - Add a case to `app/component_selector.tsx` that maps a new `component` string to that React component.
  - Update the UI schema (`public/schema.json` or the external schema at `NEXT_PUBLIC_SCHEMA`) by adding a new entry in `header.tabs` pointing to the new component and endpoint.
- To change graph coloring or edge behavior:
  - Modify `schema.nodes[*]` and `schema.edges[*]` (colors, `order`, `ring_label`, `border_color`, `edge_suffix`, `directed`), then inspect `utils/initialize.ts` and `app/api/initialize/helper.ts` to see how those fields propagate.
  - Use `GET /api/initialize` and `GET /api/knowledge_graph` responses to verify that `colors`, `aggr_scores`, and `arrow_shape` look as expected.
- To debug data issues:
  - Inspect Neo4j directly for the relevant labels and relations referenced in the schema.
  - Use the Cypher templates in the knowledge-graph API routes as a guide for manual queries.
  - For enrichment, check the raw JSON returned from Enrichr (`NEXT_PUBLIC_ENRICHR_URL`) and compare it with `POST /api/enrichment` responses.
