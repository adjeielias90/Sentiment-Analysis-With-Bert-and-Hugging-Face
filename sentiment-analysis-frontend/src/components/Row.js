import React, { useState, useEffect } from "react";
import { getReviewSentiment } from "../API/APIs";
import Happy from './images/happy.png'
import Angry from './images/angry.png';
import Nautrall from './images/nautrall.png';

import Face from './images/OIP.jpg'

function Row() {
   


    const [reviews, setReviews] = useState([]);
    const [name,setName] = useState('');
    const [text,setText]=useState('');
    

    const submitHandler = e => {

        e.preventDefault();
        const body = { review : text };
        setName('');
        setText('');
        getReviewSentiment(body).then(data => {setReviews( prevState => [...prevState, {user : name, prediction: data.prediction, review:data.review}]);console.log(reviews)});
        
    }

    
  
    
    return (
        <div className="rt-container">
        <div className="for_comment">
        <div id="respond">
          <h3>Leave A Review</h3>
          <form  onSubmit={submitHandler} id="commentForm" method="post" className="cmxform">
            <div className="commentfields">
              <label className="name">Name <span>*</span></label> 
              <input onChange={e=> setName(e.target.value)} value={name} name="name" id="cname" className="comment-input required" type="text"/>
            </div>
           
           
            <div className="commentfields">
              <label>Reviews <span>*</span></label>
              <textarea onChange={e=> setText(e.target.value)} id="ccomment" value={text} className="comment-textarea required" name="comment"></textarea>
            </div>
            <div className="commentfields">
              <input className="commentbtn" type="submit" value="Post Review"/>
            </div>
          </form>
        </div>


        </div>
        <div className="for_reviews">
        <div className="col-rt-12">
          
       <div className="content">
        <h2>(5) Watchers Comments </h2>
        {reviews.map(r => <div id="reader" style={{width:"100%"}}>
          <ol>
            <li> 
              <div className="comment_box"> <a href="#"> <img src={Face} alt="avatar"/> </a>
                <div className="inside_comment">
                  <div className="comment-meta">
                    <div className="commentsuser">{r.user || "Unknown"}</div>
                    <div className="comment_date">December 1, 2012 at 1:32 am</div>
                  </div>
                </div>


               
                <div className="comment-body">
                                
                                
                                <p>{r?.review}</p>
                                <br/>
                                    <p><h4>Prediction : {r?.prediction}</h4> </p>
                                   
                                </div>
                <div className="reply"> <img src={r.prediction ==="negative" ? Angry : r.prediction === "positive" ? Happy : Nautrall} /> </div>
                <div className="arrow-down"></div>
              </div>
                
            </li>
          </ol>
        </div>)}
        
      </div>
          
      </div>
  </div>
  </div>
      
        
    );
  }
  
  
  
  
  export default Row;