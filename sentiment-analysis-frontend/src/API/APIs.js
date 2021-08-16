
const BASE_URL = "http://c744c51b3d22.ngrok.io";

export function getReviewById(id) {
    const url = BASE_URL+"/api/review/"+id;
    return fetch(url)
    .then((response) => response.json())
    .catch((error) => console.error(error))
}  

export function getReviewSentiment(body) {
    const url = BASE_URL+"/api/review/prediction";
    
   const req = {
        method: 'POST',
        body: JSON.stringify({
            review: body.review
        }),
        headers: {
            'Content-type': 'application/json; charset=UTF-8'
        }
    }
    return fetch(url,req)
    .then((response) => response.json())
    .catch((error) => console.error(error))
}  

