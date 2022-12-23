class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector(".chatbox__button"),
            chatBox: document.querySelector(".chatbox__support"),
            sendButton: document.querySelector(".send__button"),
        };
        this.state = false;
        this.message = [];
    }

    display() {
        const { openButton, chatBox, sendButton } = this.args;
        openButton.addEventListener("click", () => this.toggleState(chatBox));
        sendButton.addEventListener("click", () => this.onSendButton(chatBox));
        const node = chatBox.querySelector("input");

        node.addEventListener("keyup", (key) => {
            if (key.key == "Enter") {
                this.onSendButton(chatBox);
            }
        });

        //console.log(this.args)
    }

    toggleState(chatBox) {
        this.state = !this.state;
        if (this.state) {
            chatBox.classList.add("chatbox--active");
        } else {
            chatbox.classList.remove("chatbox--active");
        }
    }

    onSendButton(chatbox) {
        //alert($SCRIPT_ROOT)
        var textField = chatbox.querySelector("input");
        //console.log(textField,textField.value)
        let text1 = textField.value;

        if (text1 === "") {
            return;
        }
        let msg1 = { name: "user", message: text1 };
        this.message.push(msg1);
        // console.log(text1)
        fetch($SCRIPT_ROOT + "/predict", {
            method: "POST",
            body: JSON.stringify({ message: text1 }),
            mode: "cors",
            headers: {
                "Content-Type": "application/json",
            },
        })
            .then((r) => r.json())
            .then((r) => {
                // console.log(r)
                let msg2 = { name: "Sam", message: r };
                this.message.push(msg2);
                this.updateChatText(chatbox);
                textField.value = "";
            })
            .catch((error) => {
                console.error("Error", error);
                this.updateChatText(chatbox);
                textField.value = "";
            });
    }

    updateChatText(chatbox) {
        //console.log(this.message)
        var html = "";
        this.message
            .slice()
            .reverse()
            .forEach(function (item, number) {
                if (item.name === "Sam") {
                    //console.log(item.name)
                    html +=
                        '<div class="messages__item messages__item--visitor">' +
                        item.message +
                        "</div>";
                    //console.log(html)
                } else {
                    html +=
                        '<div class="messages__item messages__item--operator">' +
                        item.message +
                        "</div>";
                }
            });

        const chatmessage = chatbox.querySelector(".chatbox__messages");
        //console.log("exsisting HTML",chatmessage.innerHTML)
        chatmessage.innerHTML = html;
    }
}

const chatbox = new Chatbox();
chatbox.display();
