document.getElementById("send-button").addEventListener("click", function () {
    var userInput = document.getElementById("user-input").value;
    var chatContainer = document.getElementById("chat-container");
         chatContainer.style.zIndex=10;//编写机器人回复的function时也需要加上这一项，否则无法看见恢复的消息
    
    if (userInput.trim() !== "") {
      sendMessage(userInput);
      document.getElementById("user-input").value = ''
    }
    else{
        alert("请输入消息！");
    }
  });

  document.getElementById("input-example1").addEventListener("click",function(){
    sendMessage("马赫数的概念是什么，它和速度因数有什么区别");
    var chatContainer = document.getElementById("chat-container");
         chatContainer.style.zIndex=10;
  });
  
  document.getElementById("input-example2").addEventListener("click",function(){
    sendMessage("二重积分计算方法");
    var chatContainer = document.getElementById("chat-container");
         chatContainer.style.zIndex=10;
  });

  function sendMessage(message) {
    var chatContainer = document.getElementById("chat-container");
    var userMessageDiv = document.createElement("div");
    userMessageDiv.classList.add("user-message");
    userMessageDiv.textContent = message;
    userMessageDiv.innerHTML = `
      <div class="message-content">${message}</div>
      <div class="avatar-container">
        <img src="images/avator.png" class="avatar" alt="user acatar">
      </div>
      
    `;
    chatContainer.appendChild(userMessageDiv);}

    function clearchatcontainer(){
        var chatContainer = document.getElementById("chat-container");
        chatContainer.innerHTML = "";
    }

    document.addEventListener('DOMContentLoaded', function () {
        var userInput = document.getElementById('user-input');
        var sendButton = document.getElementById('send-button');
        userInput.addEventListener('keydown', function (event) {
            if (event.keyCode === 13 || event.key === 'Enter') {
                event.preventDefault();
                sendButton.click();
            }
        });
    });