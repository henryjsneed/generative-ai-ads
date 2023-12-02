import AWS from 'aws-sdk';

AWS.config.update({
    region: 'us-east-1'
});

const ddb = new AWS.DynamoDB.DocumentClient();

export const handler = async (event) => {
    // Parse the incoming request
    const data = JSON.parse(event.body);

    const params = {
        TableName: process.env.DDB_TABLE_NAME,
        Item: {
            'user_id': data.user_id,
            'ad_id': data.ad_id,
            'clicked': data.clicked
        }
    };

    try {
        await ddb.put(params).promise();
        return {
            statusCode: 200,
            body: JSON.stringify({ message: 'Interaction recorded' })
        };
    } catch (err) {
        console.error('Database error:', err);
        return {
            statusCode: 500,
            body: JSON.stringify({ message: 'Failed to record interaction' })
        };
    }
};
