FROM node:8-alpine

WORKDIR /app
ADD package.json /app/package.json
RUN npm config set registry http://registry.npmjs.org
RUN npm install
ADD . /app

EXPOSE 3001
CMD ["npm", "run", "start"]
