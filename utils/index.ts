import chalk from "chalk";
// import AgentType from "langchain/agents/agent_type";
import { ChromaClient, Collection, OpenAIEmbeddingFunction } from "chromadb";
import { initializeAgentExecutor } from "langchain/agents";
import { VectorDBQAChain } from "langchain/chains";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { SerpAPI } from "langchain/tools";
import { Calculator } from "langchain/tools/calculator";
import { WebBrowser } from "langchain/tools/webbrowser";
import { Chroma } from "langchain/vectorstores/chroma";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import yaml from "js-yaml";
import fs from "fs";

export const checkForBrain = async (client: ChromaClient, embedder: OpenAIEmbeddingFunction) => {
  const collectionName = process.env.COLLECTION_NAME;
  if (!collectionName) throw new Error("COLLECTION_NAME not set in .env file.");

  const collections = await client.listCollections();
  const learningAgentCollection = collections.find((collection: Collection) => collection.name === collectionName);

  if (learningAgentCollection) {
    console.log(chalk.yellow("\nBrain found.\n"));
    return await client.getCollection(collectionName, embedder);
  } else {
    console.log(chalk.yellow("\nBrain not found. Ceating a new brain.\n"));
    const collection = await client.createCollection(collectionName, {}, embedder);
    await addTestData(collection);
    return collection;
  }
};

export const answerFromMemory = async (brain: Collection, input: string, config: Config) => {
  // const query = `You are given the following input: ${input}. You can only use the given documents - do not recall info from your own memory. If the given documents are sufficient for an accurate answer, then use them to give an accurate, detailed answer. If not, respond exactly with INSUFFICIENT_DATA.`;
  const query = `
    Find an answer to the input provided below.
    Respond in a short paragraph that reiterates the input and provides an accurate, detailed answer. 
    Include any relevant links or images in your response.

    If you are not confidfent in your answer, you should make an educated guess and simply inform the user of your uncertainty.

    --- user input ---
    ${input}
    --- end user input ---
  `;

  let memoryCount = await brain.count();
  if (memoryCount > 5) {
    memoryCount = 5;
  }

  const chromaStore = await Chroma.fromExistingCollection(new OpenAIEmbeddings(), {
    collectionName: brain.name
  });
  const chromaRes = await chromaStore.similaritySearch(input, memoryCount);

  const model = new ChatOpenAI({ temperature: config.openai.temperature.recall, maxTokens: 2000 });

  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments(chromaRes.map((doc) => doc.pageContent));

  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
  const chain = VectorDBQAChain.fromLLM(model, vectorStore);

  const res = await chain.call({
    query
  });

  return res.text;
};

export const answerFromSearch = async (brain: Collection, input: string, config: Config) => {
  const inputPlus = `
    Find an answer to the input provided below.
    Respond in a short paragraph that reiterates the input and provides an accurate, detailed answer. 
    Include any relevant links or images in your response.

    If you are not confidfent in your answer, you should make an educated guess and simply inform the user of your uncertainty.

    --- user input ---
    ${input}
    --- end user input ---
  `;

  const model = new ChatOpenAI({ temperature: config.openai.temperature.search });
  const embeddings = new OpenAIEmbeddings();

  const tools = [new WebBrowser({ model, embeddings }), new Calculator()];

  if (process.env.SERPAPI_API_KEY) {
    tools.push(new SerpAPI(process.env.SERPAPI_API_KEY, {as_sitesearch: "reddit.com" }));
  }

  const executor = await initializeAgentExecutor(tools, model, "chat-zero-shot-react-description", true);

  try {
    const result = await executor.call({ input: inputPlus });
    return result.output;
  } catch (e) {
    console.log(e);
    console.log(chalk.red("\nI made a mistake. Trying again..."));
    return answerFromSearch(brain, input, config);
  }
};

export const addMemory = async (brain: Collection, memory: string) => {
  const memoryCount = await brain.count();

  await brain.add(memoryCount.toString(), undefined, {}, memory);
};

const addTestData = async (collection: any) => {
  // add test data
  await collection.add(
    ["1", "2"],
    undefined,
    [{}, {}],
    [
      // Julius Caesar
      `reddit.com is a great source for gathering information about anything in the social zeitgeist`,

      // Alexander the Great
      `You can teach yourself any programming language by visiting github.com and searching for the language you want to learn.`
    ]
  );
};

// This function parses the YAML agent configuration from the agent configuration file (agent.yaml)
// and returns it as a Config object.
export function getConfig(): Config {
  const config = yaml.load(fs.readFileSync('agent.yaml', 'utf8'));
  return config;
}


// The type for the agent configuration.
export type Config = {
  openai: {
    model: string,
    temperature: {
      recall: number,
      search: number,
    }
  },
  langchain: {
    agent: String,
  },
};
