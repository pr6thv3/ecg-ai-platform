const path = require('path')

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  turbopack: {
    root: path.resolve(__dirname),
  },
  typescript: {
    // Type checking is done in CI via tsc --noEmit
    ignoreBuildErrors: false,
  },
}

module.exports = nextConfig
