# Serverless Inference RAG Demo

A comprehensive multi-provider RAG demo showcasing enterprise-grade AI integration with MongoDB Atlas Vector Search, DigitalOcean's Gradient AI Platform, and Cohere's advanced re-ranking capabilities for intelligent support ticket processing.

## Features

- **🏢 Multi-Provider Architecture**: Seamlessly integrates multiple AI service providers
- **📊 Vector Search**: Semantic search using MongoDB Atlas Vector Search with embeddings
- **🎯 Advanced Re-ranking**: Cohere's state-of-the-art rerank-v3.5 model for superior relevance
- **🤖 LLM Integration**: DigitalOcean Gradient AI Platform with Llama 3 models
- **🔍 Simple Comparison**: Clear before/after comparison of search quality with re-ranking
- **🎫 Support Ticket Processing**: Realistic customer support ticket data scenarios
- **💻 Interactive Demo**: Command-line interface for testing different AI functionalities
- **📝 Sample Questions**: Built-in sample questions that appear after each query for easy copy-paste
- **📁 Flexible Data Management**: JSON-based sample data for easy customization

## Prerequisites

- Python 3.8+
- **MongoDB Atlas** account with Vector Search enabled
- **DigitalOcean Gradient AI Platform** API key
- **Cohere** API key for advanced re-ranking
- **OpenAI** API key (optional, for embeddings)

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Serverless_Inference_RAG
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Create a `.env` file in the root directory:

   ```env
   DO_GENAI_KEY=your_digitalocean_gradient_ai_api_key_here
   MONGODB_URI=your_mongodb_atlas_connection_string_here
   COHERE_API_KEY=your_cohere_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Configuration

### MongoDB Atlas Vector Search Index

Create a Vector Search Index in your MongoDB Atlas collection with the following configuration:

```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1536,
        "similarity": "cosine"
      }
    }
  }
}
```

### Sample Data

The project includes sample support ticket data in the `SAMPLE_DATA` directory:

- `support_tickets.json`: Basic support ticket examples
- `extended_tickets.json`: Extended dataset with diverse ticket types

You can add your own JSON files to the `SAMPLE_DATA` directory. Each file should contain an array of ticket objects with the following structure:

```json
[
  {
    "_id": "unique_id",
    "ticket_id": "T001",
    "title": "Issue Title",
    "description": "Detailed description of the issue",
    "category": "Category Name",
    "priority": "High/Medium/Low/Critical",
    "status": "Open/In Progress/Closed"
  }
]
```

## Usage

### Method 1: Individual Scripts

1. **Data Ingestion**

   ```bash
   python 1_ingest.py
   ```

   This script will:

   - Load all JSON files from the `SAMPLE_DATA` directory
   - Generate embeddings for each ticket
   - Insert data into MongoDB Atlas
   - Display statistics about the ingested data

2. **Query Testing**

   ```bash
   python 2_query.py
   ```

3. **Test Cohere Integration**

   ```bash
   python test_cohere.py
   ```

4. **Re-ranking Comparison Demo**
   ```bash
   python 4_reranking_demo.py
   ```

### Method 2: Interactive Demo

Run the comprehensive demo with all features:

```bash
python demo.py
```

The demo provides the following options:

1. **Test Endpoint Connections**: Verify all AI service connections (MongoDB, DigitalOcean Gradient AI, Cohere, OpenAI)
2. **Data Ingestion**: Load sample data with automatic embedding generation
3. **RAG Query Demo**: Test semantic search and AI-powered answer generation
4. **Re-ranking Comparison Demo**: Compare Vector Search vs Cohere re-ranking side-by-side
5. **Display Vector Index Config**: Show MongoDB Vector Search configuration
6. **Exit**: Close the application

### Re-ranking Methods

The demo includes four different search and re-ranking approaches:

1. **Original Vector Search**: Basic semantic similarity search
2. **Semantic Re-ranking**: Re-rank results using embedding similarity
3. **LLM Re-ranking**: Use language model to intelligently re-rank results
4. **Hybrid Re-ranking**: Combine semantic and LLM re-ranking methods

## Sample Queries

The system now provides built-in sample questions that appear during interactive sessions for easy copy-paste:

1. **What login issues do we have?**
2. **What payment problems exist?**
3. **What are the high priority issues?**
4. **What mobile app crashes are reported?**
5. **What security vulnerabilities need attention?**
6. **What feature requests are pending?**
7. **What network connectivity problems exist?**
8. **What database errors are occurring?**
9. **What UI/UX issues need fixing?**
10. **What performance problems are reported?**

These sample questions will automatically appear:

- At the start of each interactive session
- After every query you submit
- Making it easy to copy and paste new questions for testing

## Project Structure

```
Serverless_Inference_RAG/
├── demo.py                 # Main interactive demo
├── 1_ingest.py            # Data ingestion script
├── 2_query.py             # Query testing script
├── 4_reranking_demo.py    # Re-ranking comparison demo
├── test_cohere.py         # Cohere API integration test
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
├── README.md             # This file
└── SAMPLE_DATA/          # Sample ticket data
    ├── support_tickets.json
    └── extended_tickets.json
```

## API Keys Setup

### DigitalOcean Gradient AI Platform

1. Sign up for [DigitalOcean Gradient AI Platform](https://cloud.digitalocean.com/gen-ai)
2. Generate an API key
3. Add it to your `.env` file as `DO_GENAI_KEY`

### MongoDB Atlas

1. Create a MongoDB Atlas cluster
2. Enable Vector Search
3. Get your connection string
4. Add it to your `.env` file as `MONGODB_URI`

### Cohere API

1. Sign up for [Cohere](https://cohere.com/)
2. Generate an API key
3. Add it to your `.env` file as `COHERE_API_KEY`

### OpenAI API

1. Create an OpenAI account
2. Generate an API key
3. Add it to your `.env` file as `OPENAI_API_KEY`

## Troubleshooting

### Common Issues

1. **403 Forbidden Error**: Check your API keys and ensure they're correctly set in the `.env` file
2. **Connection Timeout**: Verify your MongoDB Atlas connection string and network access
3. **Embedding Generation Failed**: Ensure your OpenAI API key is valid and has sufficient credits
4. **No JSON Files Found**: Make sure the `SAMPLE_DATA` directory exists and contains valid JSON files

### Debug Mode

For detailed logging, you can modify the scripts to include more verbose output or add debug print statements.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please:

1. Check the troubleshooting section
2. Review the MongoDB Atlas and DigitalOcean documentation
3. Open an issue in the repository

## Acknowledgments

- **MongoDB Atlas** for enterprise-grade Vector Search capabilities
- **DigitalOcean Gradient AI Platform** for scalable LLM inference services
- **Cohere** for state-of-the-art semantic re-ranking technology
- **OpenAI** for high-quality embedding generation
- The open-source community for various Python packages used in this project
