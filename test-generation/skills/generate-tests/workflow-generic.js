export const meta = {
  name: 'generate-tests-generic',
  description: 'Generic test generation with configurable framework, tracker, and analyzer',
  phases: [
    { title: 'Fetch', detail: 'Retrieve issue from tracker' },
    { title: 'Generate', detail: 'Create test cases' },
    { title: 'Analyze', detail: 'Impact analysis (optional)' },
    { title: 'Document', detail: 'Create documentation (optional)' },
    { title: 'Upload', detail: 'Upload results (optional)' },
  ],
}

// Parse arguments
const config = parseArguments(args);

log(`Generating tests for ${config.issueKey}`);

// Phase 1: Fetch Issue
phase('Fetch');

const issue = await agent(
  `Task: Fetch issue from ${config.tracker.type}

**Issue:** ${config.issueKey}
**Tracker:** ${config.tracker.type} (${config.tracker.url})
**Credentials:** ${config.tracker.credentials}

Fetch the issue and return:
{
  "issue_key": "${config.issueKey}",
  "summary": "issue title",
  "description": "full description",
  "labels": ["label1", "label2"],
  "affected_component": "component name from issue",
  "reproduction_steps": "steps if available"
}
`,
  {
    label: `fetch-${config.issueKey}`,
    schema: {
      type: "object",
      properties: {
        issue_key: { type: "string" },
        summary: { type: "string" },
        description: { type: "string" },
        labels: { type: "array", items: { type: "string" } },
        affected_component: { type: "string" },
        reproduction_steps: { type: "string" }
      },
      required: ["issue_key", "summary", "description"]
    }
  }
);

log(`Issue fetched: ${issue.summary}`);

// Phase 2: Generate Tests
phase('Generate');

const tests = await agent(
  `Task: Generate ${config.generation.count} tests for ${config.framework.name}

**Issue:** ${issue.summary}
**Description:** ${issue.description}
**Component:** ${issue.affected_component || 'unknown'}

**Framework Configuration:**
- Framework: ${config.framework.name}
- Source: ${config.framework.source}
- Test framework: ${config.framework.test_framework}
- Test pattern: ${config.generation.test_file_pattern}

**Generation Rules:**
1. Copy source: cp -r ${config.framework.source} ${config.generation.output_dir}/${config.issueKey}
2. Generate ${config.generation.count} test functions
3. Follow ${config.framework.test_framework} conventions
4. Include edge cases: ${config.generation.include_edge_cases}
5. Include regression: ${config.generation.include_regression}
6. Add issue annotation: # ${config.issueKey}

**Test File Selection:**
Use this logic to find the correct test file:
${config.framework.test_file_resolver || 'Auto-detect based on component name'}

**Return JSON:**
{
  "file_path": "absolute path to test file",
  "start_line": number where tests added,
  "lines_added": total lines,
  "test_count": number of tests,
  "test_class": "class name if applicable",
  "tests": [
    {
      "name": "test_function_name",
      "description": "what it tests",
      "type": "functional|edge_case|regression"
    }
  ]
}
`,
  {
    label: `generate-${config.issueKey}`,
    schema: {
      type: "object",
      properties: {
        file_path: { type: "string" },
        start_line: { type: "number" },
        lines_added: { type: "number" },
        test_count: { type: "number" },
        test_class: { type: "string" },
        tests: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              description: { type: "string" },
              type: { type: "string" }
            },
            required: ["name", "description", "type"]
          }
        }
      },
      required: ["file_path", "start_line", "lines_added", "test_count", "tests"]
    }
  }
);

log(`Generated ${tests.test_count} tests in ${tests.file_path}`);

// Phase 3: Impact Analysis (optional)
let analysis = null;

if (config.analyzer && config.analyzer.type !== 'none') {
  phase('Analyze');
  log(`Running ${config.analyzer.type} impact analysis`);

  analysis = await agent(
    `Task: ${config.analyzer.type} Impact Analysis

**Issue:** ${issue.summary}
**Description:** ${issue.description}
**Tests created:** ${tests.test_count} in ${tests.file_path}

**Analyzer Configuration:**
- Type: ${config.analyzer.type}
- Framework: ${config.framework.name}
- Trace depth: ${config.analyzer.trace_depth || 5}
- Risk levels: ${(config.analyzer.risk_levels || ['low', 'medium', 'high']).join(', ')}

**Your Task:**
Analyze the impact of this issue on the ${config.framework.name} codebase.

${getAnalyzerInstructions(config.analyzer.type, config.framework.name)}

**Return JSON:**
{
  "analyzer": "${config.analyzer.type}",
  "risk_level": "low|medium|high",
  "components_identified": ["component names"],
  "traced_flow": [
    {
      "component": "name",
      "implementation": "description",
      "file": "path/to/file",
      "line": number
    }
  ],
  "affected_modules": ["module names"],
  "existing_tests": ["test files that cover this area"],
  "recommended_tests": ["additional tests to run"]
}
`,
    {
      label: `analyze-${config.issueKey}`,
      phase: 'Analyze',
      schema: {
        type: "object",
        properties: {
          analyzer: { type: "string" },
          risk_level: { type: "string" },
          components_identified: { type: "array", items: { type: "string" } },
          traced_flow: {
            type: "array",
            items: {
              type: "object",
              properties: {
                component: { type: "string" },
                implementation: { type: "string" },
                file: { type: "string" },
                line: { type: "number" }
              },
              required: ["component", "implementation"]
            }
          },
          affected_modules: { type: "array", items: { type: "string" } },
          existing_tests: { type: "array", items: { type: "string" } },
          recommended_tests: { type: "array", items: { type: "string" } }
        },
        required: ["analyzer", "risk_level", "components_identified"]
      }
    }
  );

  log(`Analysis complete: Risk ${analysis.risk_level.toUpperCase()}`);
}

// Phase 4: Documentation (optional)
let documentation = {};

if (config.documentation && config.documentation.formats.length > 0) {
  phase('Document');
  log(`Creating documentation: ${config.documentation.formats.join(', ')}`);

  documentation = await agent(
    `Task: Create test documentation

**Issue:** ${config.issueKey} - ${issue.summary}
**Tests:** ${tests.test_count} tests created
**Analysis:** ${analysis ? \`Risk \${analysis.risk_level}\` : 'No analysis'}

**Formats to create:** ${config.documentation.formats.join(', ')}

**For each format:**

${config.documentation.formats.includes('excel') ? \`
**Excel:**
- Create: ${config.generation.output_dir}/TestCases_${config.issueKey}.xlsx
- Columns: Test ID | Scenario | Expected Behavior | Test Type | Priority
- Include all ${tests.test_count} tests
- Template: ${config.documentation.excel_template || 'default'}
\` : ''}

${config.documentation.formats.includes('markdown') ? \`
**Markdown:**
- Create: ${config.generation.output_dir}/TestCases_${config.issueKey}.md
- Sections: Summary, Tests, Analysis (if available)
- Include test details and coverage
\` : ''}

**Return paths:**
{
  "excel": "path if created",
  "markdown": "path if created"
}
`,
    {
      label: `document-${config.issueKey}`,
      schema: {
        type: "object",
        properties: {
          excel: { type: "string" },
          markdown: { type: "string" }
        }
      }
    }
  );

  log(`Documentation created`);
}

// Phase 5: Upload (optional)
if (config.upload && config.upload.enabled) {
  phase('Upload');
  log(`Uploading to ${config.tracker.type}`);

  await agent(
    `Task: Upload results to ${config.tracker.type}

**Issue:** ${config.issueKey}
**Tracker:** ${config.tracker.type} (${config.tracker.url})

${config.upload.attachments && config.upload.attachments.length > 0 ? \`
**Attachments to upload:**
${config.upload.attachments.map(fmt => documentation[fmt]).filter(Boolean).join('\\n')}
\` : ''}

${config.upload.comments && config.upload.comments.includes('tests_summary') ? \`
**Comment 1 - Test Summary:**

Test Cases Generated for ${config.issueKey}

- Framework: ${config.framework.name}
- Test file: ${tests.file_path}
- Line: ${tests.start_line}
- Tests created: ${tests.test_count}
- Lines added: ${tests.lines_added}
${tests.test_class ? \`- Test class: \${tests.test_class}\` : ''}

**Tests Created:**
${tests.tests.map((t, i) => \`\${i + 1}. \${t.name} (\${t.type}) - \${t.description}\`).join('\\n')}
\` : ''}

${config.upload.comments && config.upload.comments.includes('impact_analysis') && analysis ? \`
**Comment 2 - Impact Analysis:**

${config.analyzer.type} Impact Analysis

**Risk Level:** \${analysis.risk_level.toUpperCase()}

**Components Identified:**
\${analysis.components_identified.map((c, i) => \`\${i + 1}. \${c}\`).join('\\n')}

**Implementation Flow:**
\${analysis.traced_flow.map(t => \`- \${t.component} → \${t.implementation}\${t.file ? \` (\${t.file}:\${t.line || '?'})\` : ''}\`).join('\\n')}

**Affected Modules:** \${analysis.affected_modules.join(', ')}

**Existing Test Coverage:**
\${analysis.existing_tests.map(t => \`- \${t}\`).join('\\n')}

**Recommended Tests:**
\${analysis.recommended_tests.map(t => \`- \${t}\`).join('\\n')}
\` : ''}

**Use credentials from:** ${config.tracker.credentials}

Return "Success" when done.
`,
    { label: `upload-${config.issueKey}` }
  );

  log(`Upload complete`);
}

// Format summary
const summary = \`✅ Tests generated for ${config.issueKey}

**Framework:** ${config.framework.name}
**Issue:** ${issue.summary}

**Tests:**
- Location: ${tests.file_path} (line ${tests.start_line})
- Count: ${tests.test_count}
- Lines added: ${tests.lines_added}
${tests.test_class ? \`- Test class: \${tests.test_class}\` : ''}

**Test List:**
${tests.tests.map((t, i) => \`\${i + 1}. \${t.name} (\${t.type})\n   \${t.description}\`).join('\\n')}

${analysis ? \`
**Impact Analysis:**
- Analyzer: ${analysis.analyzer}
- Risk: ${analysis.risk_level.toUpperCase()}
- Components: ${analysis.components_identified.length}
- Affected modules: ${analysis.affected_modules.length}
\` : ''}

${Object.keys(documentation).length > 0 ? \`
**Documentation:**
${Object.entries(documentation).filter(([_, v]) => v).map(([k, v]) => \`- \${k}: \${v}\`).join('\\n')}
\` : ''}

${config.upload && config.upload.enabled ? \`
**Upload:** ✅ Results uploaded to ${config.tracker.type}
\` : ''}
\`;

return {
  issueKey: config.issueKey,
  framework: config.framework.name,
  summary,
  tests,
  analysis,
  documentation
};

// ============================================================================
// Helper Functions
// ============================================================================

function parseArguments(args) {
  // Parse command-line style arguments or use config file
  // Returns normalized configuration object

  const defaults = {
    framework: {
      name: 'pytorch',
      source: '/path/to/source',
      test_framework: 'pytest'
    },
    tracker: {
      type: 'jira',
      url: 'https://tracker.com',
      credentials: '~/.config/credentials.yaml'
    },
    generation: {
      count: '5-6',
      output_dir: '/tmp/tests',
      test_file_pattern: 'test_{component}.py',
      include_edge_cases: true,
      include_regression: true
    },
    analyzer: {
      type: 'torchtalk',
      trace_depth: 5,
      risk_levels: ['low', 'medium', 'high']
    },
    documentation: {
      formats: ['excel', 'markdown']
    },
    upload: {
      enabled: true,
      attachments: ['excel'],
      comments: ['tests_summary', 'impact_analysis']
    }
  };

  // TODO: Parse args and merge with config file
  return { ...defaults, issueKey: typeof args === 'string' ? args : args.issueKey };
}

function getAnalyzerInstructions(analyzerType, framework) {
  const instructions = {
    torchtalk: \`
Use TorchTalk to trace:
1. Identify ${framework} functions mentioned in the issue
2. Trace Python → C++ → CUDA implementations
3. Use file search to find implementations
4. Assess risk based on how critical the component is
5. Find existing tests that cover related functionality
\`,
    codeql: \`
Use CodeQL to analyze:
1. Build call graph for affected components
2. Trace data flow through the codebase
3. Identify all callers and callees
4. Find files that would be affected by changes
\`,
    custom: \`
Use custom analysis:
1. Search for function/class names in the codebase
2. Use grep to find usages
3. Manually trace through imports and calls
4. Make best-effort risk assessment
\`,
    none: \`Skip impact analysis\`
  };

  return instructions[analyzerType] || instructions.custom;
}
