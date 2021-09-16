FROM node:14-alpine
WORKDIR /backend
COPY package.json /backend/package.json
RUN npm install
COPY . /backend/
EXPOSE 3000
EXPOSE 3001
CMD ["npm", "run", "start"]