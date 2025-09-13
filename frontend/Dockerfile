# Build React app
FROM node:20-alpine AS build
WORKDIR /app

# Build context is ./frontend, so copy from current dir
COPY package.json /app/package.json
COPY vite.config.js /app/vite.config.js
COPY index.html /app/index.html
COPY src /app/src

RUN npm ci || npm install
RUN npm run build

# Serve with nginx and proxy /api to backend
FROM nginx:1.25-alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]