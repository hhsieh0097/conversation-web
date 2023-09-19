// Cache all DOM elements
var chatForm = document.querySelector('.chat-input-area');
var chatInputField = document.querySelector('.chat-input-area_input');
var chatLog = document.querySelector('.chat-log');

var text_input = document.getElementById('input-holder');
text_input.disabled = true;

var submit_button = document.getElementById('submit-button');

function accept() {
	// Send the selection to the server using Flask
	const information = document.querySelector('.information');
	information.style.display = 'none';

	const portraitContainer = document.querySelector('.portrait-container');
	portraitContainer.style.display = 'block';
}

function start() {
	var age = document.getElementById('age-holder').value;
	var gender = document.querySelector('input[name="genders"]:checked').id;

	var warning = document.querySelector('#js-script').dataset.warning;
	if (age === '' || isNaN(age) || (age <= 0)) {
		alert(warning);

		return;
	}

	console.log(age);
	console.log(gender);

	var data = {
		age: age,
		gender: gender
	};

	fetch('/save_porfile', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify(data)
	})
		.then(response => response.json())
		.then(data => {
			// Hide the gender container
			const portraitContainer = document.querySelector('.portrait-container');
			portraitContainer.style.display = 'none';

			text_input.disabled = false;
			submit_button.disabled = false;

			// First message
			var greeting = document.querySelector('#js-script').dataset.text;
			var agentMessage = getMessageElement(greeting, false);
			chatLog.append(agentMessage);
		})
		.catch(error => {
			console.error(error);
		});
}


function genderSelection(gender) {
	// Send the selection to the server using Flask
	fetch('/save_gender', {
		method: 'POST',
		body: JSON.stringify({ selection: gender }),
		headers: {
			'Content-Type': 'application/json'
		}
	})
		.then(response => response.json())
		.then(data => {
			// Hide the gender container
			const portraitContainer = document.querySelector('.portrait-container');
			portraitContainer.style.display = 'none';

			text_input.disabled = false;
			submit_button.disabled = false;

			// First message
			var greeting = document.querySelector('#js-script').dataset.text;
			var agentMessage = getMessageElement(greeting, false);
			chatLog.append(agentMessage);
		})
		.catch(error => {
			console.error('Error:', error);
		});
}

// Global loading indicator
var loading = false;

/**
 * Scrolls the contents of a container to the bottom
 * @function scrollContents 
*/
function scrollContents(container) {
	container.scrollTop = container.scrollHeight;
}

/**
 * Create a DOM element from a string value
 * @function getMessageElement 
 * @param {string} val
 * @param {boolean} isUser
 * @returns {HTMLElement}
*/
function getMessageElement(val, isUser) {
	// Create parent message element to append all inner elements
	var newMessage = document.createElement('div');

	// Add message variation class when the message belongs to the user
	if (isUser) {
		newMessage.className += 'chat-message--right ';
	}

	// Create text
	var text = document.createElement('p');
	text.append(val);
	text.className += 'chat-message_text';

	newMessage.append(text);
	newMessage.className += 'chat-message';

	return newMessage;
}

// Handle form submit (clicking on the submit button or pressing Enter)
chatForm.addEventListener('submit', function (e) {
	e.preventDefault();

	// Catch empty message. 
	if (!chatInputField.value) {
		return false;
	}

	// Add user's message to the chat log
	var userMessage = getMessageElement(chatInputField.value, true);
	chatLog.append(userMessage);

	// Scroll to last message
	scrollContents(chatLog);

	// Deal with history conversation
	text_input.readOnly = true;

	var _context = [];
	for (var i = 0; i < chatLog.children.length; i++) {
		// isUser
		if (chatLog.children[i].classList.contains('chat-message--right')) {
			_context.push({ User: chatLog.children[i].innerText });
		} else {
			_context.push({ Assistant: chatLog.children[i].innerText });
		}
	}

	// Add agent's message to the chat log
	var dialog = JSON.stringify(_context);
	$.post('/post_method', { data: dialog }, function (_data) {
		const receive_data = JSON.parse(_data);

		var agentMessage = getMessageElement(receive_data['response'], false);
		chatLog.append(agentMessage);

		console.log(receive_data['last']);
		if (receive_data['last'] != true) {
			text_input.readOnly = false;
		} else {
			text_input.placeholder = "";
		}

		scrollContents(chatLog);
	});

	if (chatInputField.value == "結束對話") {
		console.log(chatInputField)
	}

	// Clear input
	chatInputField.value = '';
});