# Stage 1: Builder
# We use a multi-stage build to compile Next.js and avoid installing devDependencies in production
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package manifests first to leverage Docker layer caching
COPY package.json package-lock.json* ./

# Install all dependencies (including devDependencies required for the build)
RUN npm ci

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
COPY package.json package-lock.json* ./
# Install ONLY production dependencies
RUN npm ci --only=production

# Copy built assets from the builder stage
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public

# Expose Next.js port
EXPOSE 3000

# Start the production server
CMD ["npm", "start"]
