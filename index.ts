import chalk from "chalk";
import { ChromaClient, Collection, OpenAIEmbeddingFunction } from "chromadb";
import * as console from "console";
import * as dotenv from "dotenv";
import * as readline from "readline";
import { addMemory, answerFromMemory, answerFromSearch, checkForBrain, getConfig, Config } from "./utils/index.js";

dotenv.config();
const agentConfig = getConfig();


(async () => {
  const client = new ChromaClient();
  const embedder = new OpenAIEmbeddingFunction(process.env.OPENAI_API_KEY);

  // await client.deleteCollection(process.env.COLLECTION_NAME); // ONLY UNCOMMENT IF YOU WANT TO RESET THE BRAIN

  const brain: Collection = await checkForBrain(client, embedder);
  console.log(chalk.yellow(`\nMemory count: ${await brain.count()}\n`));

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const learn = async (input: string, agentConfig: Config) => {
    console.log(chalk.yellow(`\nSearching...\n`));
    const searchAnswer = await answerFromSearch(brain, input, agentConfig);
    console.log(chalk.green(`${searchAnswer}\n`));
    // Review memory
    return new Promise(async (resolve, reject) => {
      rl.question(chalk.blue("\nIs this answer accurate? (y/n)\n"), async (review) => {
        if (review === "y") {
          await addMemory(brain, searchAnswer);
          console.log(chalk.yellow(`\nAdded memory!\n`));
        } else {
          console.log(chalk.yellow(`\nMemory discarded.\n`));
        }
        resolve(true);
      });

    });
  };

  const askQuestion = () => {
    return new Promise(async (resolve, reject) => { 
    rl.question(chalk.blue("\nWhat would you like to know?\n"), async (input) => {
      console.log(chalk.yellow(`\nRecalling...\n`));
      const memoryAnswer = await answerFromMemory(brain, input, agentConfig);
      console.log(chalk.yellow(`\nMemory answer: ${memoryAnswer}\n`));

      if (memoryAnswer.includes("INSUFFICIENT_DATA")) {
        await learn(input, agentConfig);
      } else {
        rl.question(chalk.blue("\nShall I search for more information? (y/n)\n"), async (review) => {
          if (review === "y") {
            await learn(input, agentConfig);
          }   
        });
      }
      resolve(true);
    });
  });
  };

  // Loop until user exits
  while (true) {
    await askQuestion();
  }

})();

