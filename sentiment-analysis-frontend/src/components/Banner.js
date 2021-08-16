import React, { useState, useEffect } from "react";
import { getReviewById } from "../API/APIs";
import Happy from './images/happy.png';
import Angry from './images/angry.png';
import Nautrall from './images/nautrall.png';
import Face from './images/OIP.jpg'
const Banner = () => {

    const [review, setReview] = useState({})
    let id = 0;
    const _setId = ({ value }) => {
        id = value;
    }


    const submitHandler = e => {

        e.preventDefault();
      
            getReviewById(id).then(data => setReview(data));
        
    }

   

    return (
        <header class="ScriptHeader" style={{ marginBottom: "60px" }}>
            <div class="rt-container">
                <div class="col-rt-12">
                    <div class="rt-heading">
                        <h1>Sentiment Analyses :  Reviews</h1>



                        <div id="respond">
                            <h3>Look for a Review by ID</h3>
                            <form onSubmit={submitHandler} id="commentForm" method="post" className="cmxform">
                                <div className="commentfields" style={{ padding: '0 30px', marginLeft: '33%' }}>
                                    <label className="name">Review ID </label>
                                    <input onChange={e => _setId(e.target)} name="name" id="cname" className="comment-input required" type="number" />

                                    <input className="commentbtn" type="submit" value="Search" />
                                </div>


                            </form>
                        </div>
                    </div>
                </div>
            </div>
{Object.keys(review).length !== 0 &&
            <div id="reader" style={{ padding: '0 200px', width:"100%"}}>
                <ol>
                    <li>
                        <div className="comment_box" > <a href="#"> <img src={Face} alt="avatar" /> </a>
                            <div className="inside_comment">
                                <div className="comment-meta">
                                    <div className="commentsuser">Unknown</div>
                                    <div className="comment_date">December 1, 2012 at 1:32 am</div>
                                </div>
                            </div>
                            <div className="comment-body">
                                
                            <br/>  <br/>
                            
                            <p>{review?.review}</p>
                            <br/>
                                <p><h4>true_sentiment : {review?.true_sentiment}</h4> </p>
                                <p><h4>Prediction : {review?.prediction}</h4> </p>
                               
                            </div>
                            <div className="reply"> <img src={review.prediction ==="negative" ? Angry : review.prediction === "positive" ? Happy : Nautrall} /> </div>

                        </div>

                    </li>
                </ol>
            </div>}

        </header>
    );


};

export default Banner;