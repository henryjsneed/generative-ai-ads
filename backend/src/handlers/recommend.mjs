import AWS from 'aws-sdk';

const sagemakerRuntime = new AWS.SageMakerRuntime({
    region: 'us-east-1',
});

export const handler = async (event) => {
    // Parse the incoming request
    const userData = JSON.parse(event.body);

    // Prepare the payload for SageMaker
    const payload = JSON.stringify({
        // Format the payload as required by your SageMaker model
        user_data: userData
    });

    const params = {
        EndpointName: 'Your-SageMaker-Endpoint-Name',
        Body: payload,
        ContentType: 'application/json', 
    };

    try {
        // Invoke the SageMaker endpoint
        const sagemakerResponse = await sagemakerRuntime.invokeEndpoint(params).promise();
        const recommendations = JSON.parse(sagemakerResponse.Body.toString('utf-8'));

        return {
            statusCode: 200,
            body: JSON.stringify({ recommendations }),
        };
    } catch (err) {
        console.error('Error calling SageMaker endpoint:', err);
        return {
            statusCode: 500,
            body: JSON.stringify({ message: 'Failed to get recommendations' }),
        };
    }
};
