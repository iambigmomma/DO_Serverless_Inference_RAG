# Sample Data Directory

This directory contains sample data files for the RAG demo system.

## Files

### `support_tickets.json`

Basic support ticket data (10 tickets) covering common categories:

- Authentication issues
- Payment problems
- Technical issues
- Mobile app problems
- Feature requests

### `extended_tickets.json`

Extended support ticket data (15 additional tickets) with more diverse scenarios:

- Security incidents
- Performance issues
- Integration problems
- Compliance requests
- Accessibility concerns

## Usage

The demo system will automatically load data from these JSON files when running data ingestion. You can:

1. **Use existing data**: Run the demo with the provided sample tickets
2. **Add your own data**: Create additional JSON files following the same structure
3. **Modify existing data**: Edit the JSON files to match your use case

## Data Structure

Each ticket should follow this structure:

```json
{
  "_id": "unique_id",
  "ticket_id": "T001",
  "title": "Issue Title",
  "description": "Detailed description of the issue",
  "category": "Category Name",
  "priority": "High|Medium|Low|Critical",
  "status": "Open|In Progress|Closed"
}
```

## Categories

The sample data includes these categories:

- Authentication
- Payment
- Technical
- Mobile
- Feature Request
- Security
- Email
- UI/UX
- Integration
- Data
- Legal
- Accessibility
- Support

## Adding Your Own Data

To add your own data:

1. Create a new JSON file in this directory
2. Follow the same structure as the existing files
3. The system will automatically detect and load all `.json` files in this directory
4. Make sure each ticket has a unique `_id` and `ticket_id`

## Notes

- All JSON files in this directory will be loaded during data ingestion
- The system generates embeddings for the combined `title + description + category` text
- Ensure your data follows the JSON array format with proper structure
