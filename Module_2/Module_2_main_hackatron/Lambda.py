# -*- coding: utf-8 -*-

# This sample demonstrates handling intents from an Alexa skill using the Alexa Skills Kit SDK for Python.
# Please visit https://alexa.design/cookbook for additional examples on implementing slots, dialog management,
# session persistence, api calls, and more.
# This sample is built using the handler classes approach in skill builder.
import logging
import csv
import requests
import io
import calendar
from datetime import datetime
import ask_sdk_core.utils as ask_utils

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.handler_input import HandlerInput

from ask_sdk_model import Response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

zodiac_traits = {
    "Aries": "Aries people are energetic, confident, and adventurous.",
    "Taurus": "Taurus people are reliable, practical, and patient.",
    "Gemini": "Gemini people are curious, adaptable, and sociable.",
    "Cancer": "Cancer people are caring, emotional, and intuitive.",
    "Leo": "Leo people are confident, creative, and natural leaders.",
    "Virgo": "Virgo people are analytical, hardworking, and detail oriented.",
    "Libra": "Libra people are balanced, diplomatic, and charming.",
    "Scorpio": "Scorpio people are passionate, determined, and resourceful.",
    "Sagittarius": "Sagittarius people are adventurous, optimistic, and independent.",
    "Capricorn": "Capricorn people are disciplined, ambitious, and practical.",
    "Aquarius": "Aquarius people are innovative, independent, and intellectual.",
    "Pisces": "Pisces people are creative, compassionate, and imaginative."
}
zodiac_lucky_color = {
    "Aries": "Red",
    "Taurus": "Green",
    "Gemini": "Yellow",
    "Cancer": "White",
    "Leo": "Gold",
    "Virgo": "Green",
    "Libra": "Pink",
    "Scorpio": "Black",
    "Sagittarius": "Purple",
    "Capricorn": "Brown",
    "Aquarius": "Blue",
    "Pisces": "Sea Green"
}
zodiac_lucky_number = {
    "Aries": 9,
    "Taurus": 6,
    "Gemini": 5,
    "Cancer": 2,
    "Leo": 1,
    "Virgo": 5,
    "Libra": 6,
    "Scorpio": 8,
    "Sagittarius": 3,
    "Capricorn": 8,
    "Aquarius": 4,
    "Pisces": 7
}
zodiac_compatibility = {
    "Aries": "Leo and Sagittarius",
    "Taurus": "Virgo and Capricorn",
    "Gemini": "Libra and Aquarius",
    "Cancer": "Scorpio and Pisces",
    "Leo": "Aries and Sagittarius",
    "Virgo": "Taurus and Capricorn",
    "Libra": "Gemini and Aquarius",
    "Scorpio": "Cancer and Pisces",
    "Sagittarius": "Aries and Leo",
    "Capricorn": "Taurus and Virgo",
    "Aquarius": "Gemini and Libra",
    "Pisces": "Cancer and Scorpio"
}
zodiac_horoscope = {
    "Aries": "Today is a great day to start something new and exciting.",
    "Taurus": "Patience and persistence will help you achieve your goals today.",
    "Gemini": "Communication will open new opportunities for you today.",
    "Cancer": "Spend time with family and loved ones for positive energy.",
    "Leo": "Your confidence will help you shine in important situations today.",
    "Virgo": "Attention to detail will bring success in your tasks today.",
    "Libra": "Balance and cooperation will lead to rewarding experiences today.",
    "Scorpio": "Trust your instincts when making important decisions today.",
    "Sagittarius": "Adventure and learning opportunities may come your way today.",
    "Capricorn": "Hard work and determination will be recognized today.",
    "Aquarius": "Creative ideas will help you solve challenging problems today.",
    "Pisces": "Your imagination and kindness will positively influence others today."
}
class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        #speak_output = "Hello! Welcome to zodic sign. Tell me your birth date to find Zodic Sign."
        #reprompt_text = "Please tell me your birth day,month and year."
        speak_output = (
                            "Welcome to Hacktron Project Guide. "
                            "I can tell you your zodiac sign from your birthday. "
                            "For example, say, My birthday is January fifth twenty fifteen."
                            "I Can recommend a book based on age, gender , language"
                            "For example Suggest a book"
                        )

        reprompt_text = (
                             "Please tell me your birthday. "
                             "For example, My birthday is August sixteenth nineteen ninety or Suggest a book"
                        )

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(reprompt_text)
                .response
        )


class CaptureZodiacSignIntentHandler(AbstractRequestHandler):
    """Handler for Hello World Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("CaptureZodiacSignIntent")(handler_input)

    def filter(self, X):
        date = X.split()
        month = date[0]
        month_as_index = list(calendar.month_abbr).index(month[:3].title())
        day = int(date[1])
        return (month_as_index,day)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        slots = handler_input.request_envelope.request.intent.slots
        year = slots["year"].value
        month = slots["month"].value
        day = slots["day"].value
        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTxoGgZLrNg5BizLJcEjGV19gIiQGhesSAYCjXMjSNvqxSYhXKrxqzfAgrRLEQJMw/pub?gid=890560460&single=true&output=csv"
        response = requests.get(url)
        csv_content = response.content
        row = csv_content.decode('utf-8').splitlines()
        rows = row[1:] # excluding the first row

        zodiac = ''
        month_as_index = list(calendar.month_abbr).index(month[:3].title())
        try:
            datetime(int(year), month_as_index, int(day))
        except ValueError:
            speak_output = (
                                f"{month} {day}, {year} is not a valid date. "
                                "Please try again with a valid birth date."
                            )

            return (
                        handler_input.response_builder
                        .speak(speak_output)
                        .ask("Please tell me your birthday again.")
                        .response
                     )

        usr_dob = (month_as_index,int(day))
        for sign in rows:
            start, end , zodiac = sign.split(',')
            if self.filter(start) <= usr_dob <= self.filter(end):
                zodiac = zodiac
                break
        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["zodiac"] = zodiac
        session_attr["conversation_state"] = "zodiac_book_offer"
        #speak_output = 'I see you were born on the {day} of {month} {year}, which means that your zodiac sign will be {zodiac}.'.format(month=month, day=day, year=year, zodiac=zodiac)
        # #speak_output = (
        #                     f"You were born on {month} {day}, {year}. "
        #                     f"Your zodiac sign is {zodiac}."
        #                 )
        # speak_output = (
        #                     f"You were born on {month} {day}, {year}. "
        #                     f"Your zodiac sign is {zodiac}. "
        #                     f"{zodiac_traits[zodiac]} "
        #                     f"Your lucky color is {zodiac_lucky_color[zodiac]}. "
        #                     f"Your lucky number is {zodiac_lucky_number[zodiac]}. "
        #                     f"Your most compatible zodiac signs are {zodiac_compatibility[zodiac]}. "
        #                     f"Today's horoscope says: {zodiac_horoscope[zodiac]} "
        #                 )                
        speak_output = f"""
                            <speak>
                            You were born on {month} {day}, {year}.
                            <break time="500ms"/>
                            
                            Your zodiac sign is
                            <emphasis level="moderate">{zodiac}</emphasis>.
                            <break time="400ms"/>
                            
                            {zodiac_traits[zodiac]}
                            <break time="400ms"/>
                            
                            Your lucky color is {zodiac_lucky_color[zodiac]}.
                            <break time="300ms"/>
                            
                            Your lucky number is {zodiac_lucky_number[zodiac]}.
                            <break time="300ms"/>
                            
                            Your most compatible zodiac signs are {zodiac_compatibility[zodiac]}.
                            <break time="400ms"/>
                            
                            Today's horoscope says:
                            <break time="300ms"/>
                            
                            {zodiac_horoscope[zodiac]}
                            <break time="300ms"/>
                            
                             Would you like a book recommendation
                             based on your zodiac personality?
                            </speak>
                            """

        # return (
        #     handler_input.response_builder
        #         .speak(speak_output)
        #         # .ask("add a reprompt if you want to keep the session open for the user to respond")
        #         .response
        # )
        return (
             handler_input.response_builder
        .speak(speak_output)
        .ask("Would you like to try another Zodiac Sign or book recommendation ?"
              "Example My birthday is august sixteenth nineteen ninety,"
               "suggest a mystery book for adult in english")
        .response
        )

# class ZodiacBookRecommendationIntentHandler(AbstractRequestHandler):

#     def can_handle(self, handler_input):
#         return ask_utils.is_intent_name(
#             "ZodiacBookRecommendationIntent"
#         )(handler_input)

#     def handle(self, handler_input):
#         session_attr = handler_input.attributes_manager.session_attributes
#         state = session_attr.get("conversation_state")

#         # User said YES after "continue exploring"
#         if state == "continue_exploring":

#             session_attr.pop("conversation_state", None)

#             speak_output = (
#             "Great! Would you like to know another zodiac sign "
#             "or get a book recommendation?"
#                         )

#         return (
#             handler_input.response_builder
#                 .speak(speak_output)
#                 .ask(
#                     "You can say another zodiac sign, "
#                     "or ask for a book recommendation."
#                 )
#                 .response
#         )

#     # User said YES to zodiac-based book recommendation
#     if state == "zodiac_book_offer":
#         session_attr["conversation_state"] = "continue_exploring"
#     print(zodiac)
#     session_attr = handler_input.attributes_manager.session_attributes
#     zodiac = session_attr.get("zodiac")

#     if not zodiac:
#         speak_output = (
#             "Please tell me your birth date first so I can "
#             "find your zodiac sign before recommending a book."
#         )

#         return (
#             handler_input.response_builder
#                 .speak(speak_output)
#                 .ask(speak_output)
#                 .response
#         )

#     url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSMXV-Oa9oLhaGiopbg-VTLA7HIAWRnkAmH1DexZCUf4aegTh7ByokxXWe6YT3M0R9ctLicVicwL3LM/pub?gid=0&single=true&output=csv"

#     response = requests.get(url)
#     csv_content = response.content.decode("utf-8").splitlines()

#     rows = csv_content[1:]

#     speak_output = (
#         f"Sorry, I could not find a recommendation for {zodiac}."
#     )

#     for row in rows:

#         z, genre, book, author, rating, reading_time, difficulty = row.split(",")

#         if z.lower() == zodiac.lower():

#             speak_output = (
#                 f"{zodiac}s often enjoy {genre} stories. "
#                 f"I recommend {book} by {author}. "
#                 f"It has a rating of {rating} out of 5. "
#                 f"The estimated reading time is {reading_time}. "
#                 f"The difficulty level is {difficulty}. "
#                 f"Enjoy your reading. "
#                 f"Would you like to continue exploring Zodiac Guide?"
#             )

#             break

#     return (
#         handler_input.response_builder
#             .speak(speak_output)
#             .ask(
#                 "You can say yes to continue exploring, "
#                 "or no to exit."
#             )
#             .response
#     )
class ZodiacBookRecommendationIntentHandler(AbstractRequestHandler):

    def can_handle(self, handler_input):
        return ask_utils.is_intent_name(
            "ZodiacBookRecommendationIntent"
        )(handler_input)

    def handle(self, handler_input):

        session_attr = (
            handler_input.attributes_manager.session_attributes
        )

        state = session_attr.get("conversation_state")

        # User said YES after continue exploring
        if state == "continue_exploring":

            session_attr.pop("conversation_state", None)

            speak_output = (
                "Great! Would you like to know another zodiac sign "
                "or get a book recommendation?"
            )

            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask(
                        "You can say my birthday is August sixteenth "
                        "nineteen ninety, or suggest a mystery book "
                        "for adults in English."
                    )
                    .response
            )

        # User said YES after zodiac profile
        if state == "zodiac_book_offer":
            session_attr["conversation_state"] = (
                "continue_exploring"
            )

        zodiac = session_attr.get("zodiac")

        if not zodiac:

            speak_output = (
                "Please tell me your birth date first so I can "
                "find your zodiac sign before recommending a book."
            )

            return (
                handler_input.response_builder
                    .speak(speak_output)
                    .ask(speak_output)
                    .response
            )

        url = (
            "https://docs.google.com/spreadsheets/d/e/"
            "2PACX-1vSMXV-Oa9oLhaGiopbg-VTLA7HIAWRnkAmH1DexZCUf4aegTh7ByokxXWe6YT3M0R9ctLicVicwL3LM/"
            "pub?gid=0&single=true&output=csv"
        )

        response = requests.get(url)

        csv_content = (
            response.content.decode("utf-8").splitlines()
        )

        rows = csv_content[1:]

        speak_output = (
            f"Sorry, I could not find a recommendation "
            f"for {zodiac}."
        )

        for row in rows:

            parts = row.split(",")

            if len(parts) != 7:
                continue

            (
                z,
                genre,
                book,
                author,
                rating,
                reading_time,
                difficulty
            ) = parts

            if z.lower() == zodiac.lower():

                speak_output = (
                    f"{zodiac}s often enjoy {genre} stories. "
                    f"I recommend {book} by {author}. "
                    f"It has a rating of {rating} out of 5. "
                    f"The estimated reading time is "
                    f"{reading_time}. "
                    f"The difficulty level is "
                    f"{difficulty}. "
                    f"Enjoy your reading. "
                    f"Would you like to continue exploring "
                    f"Zodiac Guide?"
                )

                break

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(
                    "Would you like to continue exploring "
                    "Zodiac Guide?"
                )
                .response
        )
class BookRecommendationIntentHandler(AbstractRequestHandler):

    def can_handle(self, handler_input):
        return ask_utils.is_intent_name(
            "BookRecommendationIntent"
        )(handler_input)

    def handle(self, handler_input):

        slots = handler_input.request_envelope.request.intent.slots

        genre = slots["genre"].value
        age_group = slots["age_group"].value
        language = slots["language"].value

        url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRhV-vPTtngafoWq9sbvx1nU7MjRG_h_8tXpVWeHbck2CKUA86gaFEdBFhtRr59uSMZXJNeSEBXW0Zs/pub?gid=0&single=true&output=csv"

        response = requests.get(url)
        csv_content = response.content.decode('utf-8').splitlines()

        rows = csv.reader(csv_content)

        next(rows)  # skip header

        book_found = False

        for row in rows:
            csv_genre = row[0]
            csv_age = row[1]
            csv_language = row[2]
            book_name = row[3]
            author = row[4]

            if (csv_genre.lower() == genre.lower() and
                csv_age.lower() == age_group.lower() and
                csv_language.lower() == language.lower()):

                speak_output = (
                    f"I recommend {book_name} by {author}."
                )

                book_found = True
                break

        if not book_found:
            speak_output = (
                f"Sorry, I could not find a "
                f"{genre} book for {age_group} "
                f"readers in {language}."
            )

        return (
            handler_input.response_builder
                .speak(speak_output+ "Would you like to try another Zodiac Sign or book recommendation ?")
                .ask("Would you like to try another Zodiac Sign or book recommendation ?"
                      "Example My birthday is august sixteenth nineteen ninety,"
                      "suggest a adventure book for adult in english")
                .response
        )
        
# class NoIntentHandler(AbstractRequestHandler):

#     def can_handle(self, handler_input):
#         return ask_utils.is_intent_name(
#             "AMAZON.NoIntent"
#         )(handler_input)

#     # def handle(self, handler_input):

#     #     speak_output = (
#     #         "That's okay. You can always ask me for "
#     #         "personalized book recommendations or "
#     #         "zodiac information anytime. "
#     #         "Have a wonderful day!"
#     #     )

#     #     return (
#     #         handler_input.response_builder
#     #             .speak(speak_output)
#     #             .set_should_end_session(True)
#     #             .response
#     #     )
#     def handle(self, handler_input):

#         session_attr = handler_input.attributes_manager.session_attributes
#         state = session_attr.get("conversation_state")

#         if state == "continue_exploring":
#             session_attr.clear()
    
#             speak_output = (
#                 "Thank you for using Zodiac Guide. "
#                 "Have a wonderful day!"
#                           )

#         else:
#             session_attr.clear()
    
#             speak_output = (
#                 "That's okay. You can always ask me for personalized "
#                 "book recommendations or zodiac information anytime. "
#                 "Have a wonderful day!"
#                           )

#         return (
#             handler_input.response_builder
#                 .speak(speak_output)
#                 .set_should_end_session(True)
#                 .response
#               )
class NoIntentHandler(AbstractRequestHandler):

    def can_handle(self, handler_input):
        return ask_utils.is_intent_name(
            "AMAZON.NoIntent"
        )(handler_input)

    def handle(self, handler_input):

        session_attr = handler_input.attributes_manager.session_attributes
        state = session_attr.get("conversation_state")

        if state == "continue_exploring":

            session_attr.clear()

            speak_output = (
                "Thank you for using Zodiac Guide. "
                "Have a wonderful day!"
            )

        else:

            session_attr.clear()

            speak_output = (
                "That's okay. You can always ask me for personalized "
                "book recommendations or zodiac information anytime. "
                "Have a wonderful day!"
            )

        return (
            handler_input.response_builder
                .speak(speak_output)
                .set_should_end_session(True)
                .response
        )
class HelpIntentHandler(AbstractRequestHandler):
    """Handler for Help Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        #speak_output = "You can say hello to me! How can I help?"
        speak_output = (
                        "You can tell me your birth date and I will find your zodiac sign. "
                        "For example, say: My birthday is January fifth twenty fifteen."
                        )

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output =( "Thank you for using Zodiac Guide. "
            "Have a wonderful day!")

        return (
            handler_input.response_builder
                .speak(speak_output)
                .response
        )

class FallbackIntentHandler(AbstractRequestHandler):
    """Single handler for Fallback Intent."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        logger.info("In FallbackIntentHandler")
        #speech = "Hmm, I'm not sure. You can say Hello or Help. What would you like to do?"
        #reprompt = "I didn't catch that. What can I help you with?"
        speech = (
                      "Sorry, I didn't understand that. "
                      "Please tell me your birthday. "
                        "For example, say: My birthday is August sixteenth nineteen ninety."
                    )

        reprompt = (
                "Please tell me your birth date to find your zodiac sign."
                    )

        return handler_input.response_builder.speak(speech).ask(reprompt).response

class SessionEndedRequestHandler(AbstractRequestHandler):
    """Handler for Session End."""
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        # Any cleanup logic goes here.

        return handler_input.response_builder.response


class IntentReflectorHandler(AbstractRequestHandler):
    """The intent reflector is used for interaction model testing and debugging.
    It will simply repeat the intent the user said. You can create custom handlers
    for your intents by defining them above, then also adding them to the request
    handler chain below.
    """
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("IntentRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        intent_name = ask_utils.get_intent_name(handler_input)
        speak_output = "You just triggered " + intent_name + "."

        return (
            handler_input.response_builder
                .speak(speak_output)
                # .ask("add a reprompt if you want to keep the session open for the user to respond")
                .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors. If you receive an error
    stating the request handler chain is not found, you have not implemented a handler for
    the intent being invoked or included it in the skill builder below.
    """
    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)

        speak_output = "Sorry, I had trouble doing what you asked. Please try again."

        return (
            handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
        )

# The SkillBuilder object acts as the entry point for your skill, routing all request and response
# payloads to the handlers above. Make sure any new handlers or interceptors you've
# defined are included below. The order matters - they're processed top to bottom.


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(CaptureZodiacSignIntentHandler())
sb.add_request_handler(ZodiacBookRecommendationIntentHandler())
sb.add_request_handler(BookRecommendationIntentHandler())
sb.add_request_handler(NoIntentHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())
sb.add_request_handler(IntentReflectorHandler()) # make sure IntentReflectorHandler is last so it doesn't override your custom intent handlers

sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()