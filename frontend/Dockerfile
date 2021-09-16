FROM node

RUN npm install -g http-server

WORKDIR /frontend/

COPY package*.json /frontend/

RUN npm install

COPY . /frontend/

RUN npm run build

EXPOSE 8080
CMD ["http-server", "dist/frontend"]