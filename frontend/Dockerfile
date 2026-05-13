# Stage 1: Builder
# We use a multi-stage build to compile Next.js and avoid installing devDependencies in production
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package manifests first to leverage Docker layer caching
COPY package.json package-lock.json* ./

# Install all dependencies (including devDependencies required for the build)
RUN npm install --legacy-peer-deps

# Copy the rest of the application code
COPY . .

# Build the Next.js application for production
RUN npm run build

# Stage 2: Runner (Production Image)
FROM node:18-alpine AS runner

WORKDIR /app

# Set environment to production to optimize Node.js performance
ENV NODE_ENV=production
# Disable telemetry for privacy
ENV NEXT_TELEMETRY_DISABLED=1

# Copy only the necessary files for a smaller image footprint
# Copy standalone output from the builder stage
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# Expose Next.js port
EXPOSE 3000

# Start the standalone server (no npm needed)
CMD ["node", "server.js"]
