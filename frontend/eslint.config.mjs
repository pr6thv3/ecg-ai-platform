import nextVitals from 'eslint-config-next/core-web-vitals'

const testGlobals = {
  afterEach: 'readonly',
  beforeEach: 'readonly',
  describe: 'readonly',
  expect: 'readonly',
  global: 'readonly',
  it: 'readonly',
  jest: 'readonly',
}

const config = [
  ...nextVitals,
  {
    files: ['**/*.test.ts', '**/*.test.tsx', 'tests/**/*.ts', 'tests/**/*.tsx', '__tests__/**/*.ts', '__tests__/**/*.tsx'],
    languageOptions: {
      globals: testGlobals,
    },
  },
  {
    ignores: ['coverage/**', 'tsconfig.tsbuildinfo'],
  },
]

export default config
