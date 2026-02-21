FROM node:22-alpine

WORKDIR /webapp

COPY webapp/package*.json /webapp/

RUN npm ci

COPY webapp /webapp

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
