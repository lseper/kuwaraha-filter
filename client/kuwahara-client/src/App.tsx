import Stack from "@mui/material/Stack";
import Box from "@mui/material/Box";
import Container from "@mui/material/Container";
import ArrowRightAltIcon from "@mui/icons-material/ArrowRightAlt";
import UploadCard from "./UploadCard";

function App() {
	return (
		<Container>
			<h1>The Kuwahara Filter</h1>
			<Stack direction="row" sx={{ paddingBottom: 8 }}>
				<img
					src="./imgs/turing-test.jpg"
					style={{ width: 300 }}
					alt=""
				/>

				<Box
					sx={{
						display: "flex",
						flexDirection: "row",
						justifyContent: "center",
						alignItems: "center",
					}}
				>
					<ArrowRightAltIcon fontSize="large" />
				</Box>

				<img
					src="./imgs/turing-test.jpg_kuwahara_gaussian_r20.jpg"
					style={{ width: 300 }}
					alt=""
				/>
			</Stack>
			<UploadCard />
		</Container>
	);
}

export default App;
