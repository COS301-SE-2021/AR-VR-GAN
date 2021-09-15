FROM node:14-alpine
WORKDIR /backend
COPY package.json /backend/package.json
RUN npm install
COPY . .
CMD ["npm", "run", "start"]